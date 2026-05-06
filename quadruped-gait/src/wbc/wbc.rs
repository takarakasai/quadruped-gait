//! Top-level WBC orchestrator: assemble the 3-priority Hierarchical QP
//! from the per-task formulators and decode the solution back into
//! `(q̈, f_GRF, τ)`.
//!
//! ```text
//! Task 0 (hard):   floating_base_eom + torque_limits + friction_cone + no_contact_motion
//! Task 1 (soft):   base_accel + swing_leg
//! Task 2 (soft):   contact_force
//! ```
//!
//! Mirrors `legged_control`'s `Wbc::update` line-by-line so the
//! battle-tested formulation is preserved.

use nalgebra::{DMatrix, DVector};

use super::ho_qp::WarmStart;
use super::tasks::{
    base_accel, contact_force, floating_base_eom, friction_cone, no_contact_motion,
    swing_leg, tau_gravity, torque_limits,
};
use super::{HoQp, WbcDims};

/// All inputs the WBC needs to assemble + solve one tick's QP.
///
/// The host (typically [`crate::MpcGaitController`] + [`crate::ChampGaitController`]
/// helpers) computes these from the current MuJoCo / state-estimator
/// state plus the MPC's reference output.
pub struct WbcInputs<'a> {
    pub dims: WbcDims,
    /// Mass matrix `M(q) ∈ R^(nv × nv)` from `misarta::crba`.
    pub mass: &'a DMatrix<f64>,
    /// Non-linear effects `h(q, q̇) = C·v + g ∈ R^nv`
    /// from `misarta::nonlinear_effects`.
    pub nle: &'a DVector<f64>,
    /// Stacked **linear** part of foot Jacobians: top 3 rows of the
    /// 6×nv Jacobian per foot, in world frame. Shape `(3·nc) × nv`.
    pub j_contact: &'a DMatrix<f64>,
    /// `dJ/dt · v` for each foot's linear part. Length `3·nc`.
    pub dj_v: &'a DVector<f64>,
    pub contact_flag: [bool; 4],
    pub friction_mu: f64,
    /// Per-actuator torque limit (length `na`).
    pub torque_max: &'a DVector<f64>,
    /// Reference base acceleration from MPC (6: linear + angular,
    /// world frame).
    pub a_base_des: &'a DVector<f64>,
    /// Reference Cartesian swing-foot acceleration in world frame
    /// (length `3·nc`; stance entries are ignored).
    pub a_swing_des: &'a DVector<f64>,
    /// MPC-predicted GRF (length `3·nc`, world frame).
    pub f_grf_des: &'a DVector<f64>,
    /// Static gravity-compensation torque per actuator (length `na`).
    /// Used as a soft τ reference at the lowest priority so the QP
    /// can't collapse to τ ≈ 0 — a failure mode observed when the
    /// EoM constraint allowed (f balancing gravity, τ = 0) as a
    /// feasible solution. Compute via `misarta::rnea::compute_gravity`
    /// at the current `q` (with `q̇ = 0`, `q̈ = 0`) and project to
    /// the actuated rows. Pass an all-zero vector to disable.
    pub tau_gravity: &'a DVector<f64>,
}

/// Decoded WBC solution.
#[derive(Clone, Debug)]
pub struct WbcSolution {
    pub q_ddot: DVector<f64>,
    pub f_grf: DVector<f64>,
    pub tau: DVector<f64>,
    /// Full decision-space solution `x = [q̈; f_GRF; τ]` (length
    /// `nv + 3·nc + na`). The host caches this and feeds it back as
    /// the next tick's [`WbcWarmStart::x_prev`] to dampen QP jitter.
    pub x_full: DVector<f64>,
}

impl WbcSolution {
    fn from_x(x: &DVector<f64>, dims: WbcDims) -> Self {
        Self {
            q_ddot: x.rows(dims.q_offset(), dims.nv).into_owned(),
            f_grf: x.rows(dims.f_offset(), 3 * dims.nc).into_owned(),
            tau: x.rows(dims.tau_offset(), dims.na).into_owned(),
            x_full: x.clone(),
        }
    }
}

/// Warm-start hint carried across WBC ticks.
///
/// The host (typically [`crate::WbcPipeline`]) caches the previous
/// tick's [`WbcSolution::x_full`] and feeds it back here. Each
/// priority level then adds a `(prox_weight / 2)·‖v − v_target‖²` term
/// where `v_target = prev.zᵀ · (x_prev − prev.x)` — the full-space
/// anchor reprojected into that level's null-space basis. Anchors
/// stay valid even when `prev.z` rotates between ticks (which it does:
/// it's built from a `q`-dependent equality matrix).
///
/// Single shared `prox_weight` across levels — the per-level
/// objectives are already on comparable (residual²) units and one knob
/// is easier to tune.
#[derive(Clone, Debug, Default)]
pub struct WbcWarmStart<'a> {
    /// Full-space anchor (length `nv + 3·nc + na`). `None` = cold.
    pub x_prev: Option<&'a DVector<f64>>,
    /// Proximal regularisation weight. 0.0 = no anchoring (cold).
    /// Recommended starting value: ~1e-3.
    pub prox_weight: f64,
}

/// Assemble + solve one tick of the hierarchical WBC (cold start).
///
/// Equivalent to [`solve_warm`] with [`WbcWarmStart::default()`].
pub fn solve(inputs: &WbcInputs) -> WbcSolution {
    solve_warm(inputs, &WbcWarmStart::default())
}

/// Like [`solve`] but with a [`WbcWarmStart`] hint to dampen tick-to-
/// tick jitter.
pub fn solve_warm(inputs: &WbcInputs, warm: &WbcWarmStart<'_>) -> WbcSolution {
    let dims = inputs.dims;

    // ── Per-task LSQ weights (Phase 1.4) ──────────────────────────
    // The HoQp inner cost ½‖A·v − b‖² stacks every task at the same
    // priority into one matrix; without per-task weighting, every row
    // gets equal vote regardless of physical importance. We bias each
    // task explicitly so the most important physics wins when same-
    // priority tasks conflict.
    //
    // Priority 0 (hard constraints): EoM and no_contact_motion are
    // both physically mandatory. Boosting them strongly inside the
    // priority-0 block makes the QP reach near-zero residual on
    // those rows, leaving torque-limit / friction-cone slack to
    // absorb only what's mathematically infeasible.
    const W_FLOATING_BASE_EOM: f64 = 1000.0;
    const W_NO_CONTACT_MOTION: f64 = 1000.0;
    // Priority 1 (motion tracking): the base-accel reference comes
    // from `predicted_base_accel_world` (the MPC's GRF run through
    // SRBD physics) and is the primary mechanism keeping the trunk
    // up — heavily weighted so it dominates the priority-1 LSQ.
    //
    // swing_leg is intentionally low-weight (1.0) because the WBC's
    // a_swing_des (Cartesian PD on body-frame foot targets) is a
    // different representation than the gait controller's joint-space
    // q* tracked by Position-PD. legged_control sources both from
    // OCS2's predicted joint state, but our SRBD MPC doesn't produce
    // joint-level references, so Position-PD ends up driving the
    // swing in joint space and WBC's swing_leg adds dynamic
    // compensation in Cartesian space. Keep the weight modest so the
    // two paths cooperate without one fighting the other; bumping
    // either swing_leg to 0 or up to 100 both empirically degrade
    // forward locomotion.
    const W_BASE_ACCEL: f64 = 200.0;
    const W_SWING_LEG: f64 = 1.0;
    // Priority 2 (regularisation): contact_force tracks MPC's GRF
    // prediction. Higher weight tightens the WBC's f_GRF to the MPC's
    // predicted values, which matters during trot stance windows
    // (~0.2 s) — if the WBC under-applies the predicted f_z the body
    // drops a few mm per cycle and accumulates downward drift over
    // the walking window. tau_gravity anchors τ near static gravity-
    // comp so the τ block doesn't collapse to zero in degenerate
    // null-space directions.
    const W_CONTACT_FORCE: f64 = 1.0;
    const W_TAU_GRAVITY: f64 = 5.0;

    // ── Priority 0: hard constraints ───────────────────────────────
    let task_0 = floating_base_eom::formulate(
        dims,
        inputs.mass,
        inputs.nle,
        inputs.j_contact,
    )
    .weight(W_FLOATING_BASE_EOM)
        + torque_limits::formulate(dims, inputs.torque_max)
        + friction_cone::formulate(dims, inputs.contact_flag, inputs.friction_mu)
        + no_contact_motion::formulate(
            dims,
            inputs.j_contact,
            inputs.dj_v,
            inputs.contact_flag,
        )
        .weight(W_NO_CONTACT_MOTION);

    // ── Priority 1: motion tracking ────────────────────────────────
    let task_1 = base_accel::formulate(dims, inputs.a_base_des).weight(W_BASE_ACCEL)
        + swing_leg::formulate(
            dims,
            inputs.j_contact,
            inputs.dj_v,
            inputs.a_swing_des,
            inputs.contact_flag,
        )
        .weight(W_SWING_LEG);

    // ── Priority 2: GRF + τ_grav joint regularisation ─────────────
    let task_2 = contact_force::formulate(dims, inputs.f_grf_des).weight(W_CONTACT_FORCE)
        + tau_gravity::formulate(dims, inputs.tau_gravity).weight(W_TAU_GRAVITY);

    let warm_inner = WarmStart {
        x_prev: warm.x_prev,
        prox_weight: warm.prox_weight,
    };
    let l0 = HoQp::new_with_higher_warm(task_0, None, &warm_inner);
    let l1 = HoQp::new_with_higher_warm(task_1, Some(&l0), &warm_inner);
    let l2 = HoQp::new_with_higher_warm(task_2, Some(&l1), &warm_inner);

    WbcSolution::from_x(l2.solution(), dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end sanity check: a hover scenario (4 feet on the ground,
    /// gravity holding the body up) should produce GRFs that sum to
    /// `m·g` and torques within bounds. We use a heavily simplified
    /// "robot" — just `M = m·I_(nv)` with `nv = 6 + 4`, decoupled — to
    /// keep the test independent of misarta's URDF parsing.
    ///
    /// This is more of a smoke test than a physics test; the per-task
    /// unit tests already verify the matrix layout is correct.
    #[test]
    fn hover_smoke_test_runs_without_panic() {
        // Tiny synthetic robot: 6 base DoF + 4 actuated (one per leg).
        let dims = WbcDims { nv: 10, nc: 4, na: 4 };
        let mass = DMatrix::identity(10, 10) * 5.0; // 5 kg (toy)
        // Gravity-only nle: only z-component of base.
        let mut nle = DVector::zeros(10);
        nle[2] = 5.0 * 9.81; // m * g pulling the body down
        // Each foot contributes a single +z column at the base linear z.
        let mut j = DMatrix::zeros(12, 10);
        for i in 0..4 {
            // f_i acts on base linear z only — minimal kinematics.
            j[(3 * i + 2, 2)] = 1.0;
        }
        let dj_v = DVector::zeros(12);
        let torque_max = DVector::from_vec(vec![100.0; 4]);
        // Base accel reference: hold still (zero accel).
        let a_base_des = DVector::zeros(6);
        // Stance, no swing motion.
        let a_swing_des = DVector::zeros(12);
        // MPC says: distribute weight evenly across 4 feet.
        let mut f_grf_des = DVector::zeros(12);
        for i in 0..4 {
            f_grf_des[3 * i + 2] = 5.0 * 9.81 / 4.0;
        }
        // Smoke test: τ_grav reference is irrelevant for the symbolic
        // hover scenario (decoupled M, J on a synthetic robot), so
        // pass zero to verify the new task plumbing doesn't break
        // the existing hover.
        let tau_gravity = DVector::zeros(4);

        let inputs = WbcInputs {
            dims,
            mass: &mass,
            nle: &nle,
            j_contact: &j,
            dj_v: &dj_v,
            contact_flag: [true; 4],
            friction_mu: 0.5,
            torque_max: &torque_max,
            a_base_des: &a_base_des,
            a_swing_des: &a_swing_des,
            f_grf_des: &f_grf_des,
            tau_gravity: &tau_gravity,
        };

        let sol = solve(&inputs);
        // Sanity: total normal force should approximately balance gravity.
        let total_fz: f64 = (0..4).map(|i| sol.f_grf[3 * i + 2]).sum();
        assert!(
            (total_fz - 5.0 * 9.81).abs() < 5.0,
            "Σf_z = {total_fz} should ≈ m·g = {}",
            5.0 * 9.81
        );
        // Torques bounded by limits.
        for i in 0..dims.na {
            assert!(
                sol.tau[i].abs() <= torque_max[i] + 1e-3,
                "τ[{i}] = {} exceeds limit {}",
                sol.tau[i],
                torque_max[i]
            );
        }
    }
}
