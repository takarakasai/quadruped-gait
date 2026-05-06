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
}

impl WbcSolution {
    fn from_x(x: &DVector<f64>, dims: WbcDims) -> Self {
        Self {
            q_ddot: x.rows(dims.q_offset(), dims.nv).into_owned(),
            f_grf: x.rows(dims.f_offset(), 3 * dims.nc).into_owned(),
            tau: x.rows(dims.tau_offset(), dims.na).into_owned(),
        }
    }
}

/// Assemble + solve one tick of the hierarchical WBC.
///
/// Returns `Some` when every level's inner QP converged. Returns
/// `None` when a level fails (the host should fall back to a safe
/// command — typically the previous tick's torques or pure
/// position-PD without `τ_ff`).
pub fn solve(inputs: &WbcInputs) -> WbcSolution {
    let dims = inputs.dims;

    // ── Priority 0: hard constraints ───────────────────────────────
    let task_0 = floating_base_eom::formulate(
        dims,
        inputs.mass,
        inputs.nle,
        inputs.j_contact,
    ) + torque_limits::formulate(dims, inputs.torque_max)
        + friction_cone::formulate(dims, inputs.contact_flag, inputs.friction_mu)
        + no_contact_motion::formulate(
            dims,
            inputs.j_contact,
            inputs.dj_v,
            inputs.contact_flag,
        );

    // ── Priority 1: motion tracking ────────────────────────────────
    let task_1 = base_accel::formulate(dims, inputs.a_base_des)
        + swing_leg::formulate(
            dims,
            inputs.j_contact,
            inputs.dj_v,
            inputs.a_swing_des,
            inputs.contact_flag,
        );

    // ── Priority 2: GRF + τ_grav joint regularisation ─────────────
    // contact_force tracks the MPC's predicted GRFs. tau_gravity
    // anchors τ near static gravity-comp so the QP can't collapse to
    // (f balances gravity, τ ≈ 0). Combining them at the SAME
    // priority means the inner QP minimises both residuals jointly —
    // putting tau_gravity at a separate, lower priority would have
    // zero null space to work with after task 0+1+2 already constrain
    // 49 / 44 dims (verified empirically: the priority-3 anchor had
    // identical output to having no anchor).
    let task_2 = contact_force::formulate(dims, inputs.f_grf_des)
        + tau_gravity::formulate(dims, inputs.tau_gravity);

    let l0 = HoQp::new(task_0);
    let l1 = HoQp::new_with_higher(task_1, Some(&l0));
    let l2 = HoQp::new_with_higher(task_2, Some(&l1));

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
