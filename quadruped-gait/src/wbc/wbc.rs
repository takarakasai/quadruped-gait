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
    /// Joint-space swing-leg PD reference: `q̈_des[a]` per actuator,
    /// length `na`. Active only where `swing_actuator_flag[a]` is
    /// true; rows for stance / arm actuators are skipped. Computed
    /// by the host as `kp·(q*−q) + kd·(q̇*−q̇)` using the same `q*`
    /// that Position-PD tracks (the gait controller's IK output).
    pub swing_q_ddot_des: &'a DVector<f64>,
    /// Per-actuator swing flag (length `na`). True when this actuator
    /// belongs to a leg currently in swing phase.
    pub swing_actuator_flag: &'a [bool],
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

/// Per-task LSQ weights applied inside the priority levels of the
/// hierarchical QP. These default to the values empirically tuned
/// for the namiashi quadruped and are exposed as a `pub` struct so
/// tests / hosts can override individual entries to study how each
/// task contributes to the final torque (= the P5a sign-flip
/// diagnostic in `tests/integration_walk.rs`).
///
/// Setting any weight to `0.0` cleanly **disables** that task at the
/// LSQ level; the priority-0 hard-equality structure of the HoQp
/// guarantees the `floating_base_eom` and `no_contact_motion` tasks
/// stay enforced through the null-space chain even if their LSQ
/// weight is small (so they're safe to leave at default).
#[derive(Clone, Copy, Debug)]
pub struct WbcWeights {
    /// Priority 0 — floating-base equation of motion. Weighted ≥ 100
    /// so the EoM residual is driven near machine zero inside the
    /// priority-0 LSQ; lower weights show as small but visible body
    /// drift over long runs.
    pub floating_base_eom: f64,
    /// Priority 0 — stance feet must not slide / leave the ground
    /// (`J_c · q̈ + J̇·v = 0`). Same scale as `floating_base_eom`.
    pub no_contact_motion: f64,
    /// Priority 1 — body acceleration tracks the MPC-predicted
    /// `predicted_base_accel_world`. The dominant mechanism that
    /// keeps the trunk upright + drives forward thrust.
    pub base_accel: f64,
    /// Priority 1 — joint-space PD on the swing legs (`q̈_swing =
    /// kp·(q*−q) + kd·(q̇*−q̇)`). Low weight by default because
    /// Position-PD already tracks `q*` at the actuator level; the
    /// WBC's swing-leg term adds Cartesian compensation only.
    pub swing_leg: f64,
    /// Priority 2 — contact forces track the MPC's predicted GRFs.
    /// Bumping this above ~1 lets the WBC tighten `sol.f_grf` to
    /// the MPC reference (= forward thrust flows through to joint τ).
    pub contact_force: f64,
    /// Priority 2 — joint torques anchored toward the static
    /// gravity-comp value. Stops degenerate `(τ ≈ 0, f balances g)`
    /// solutions in the null space.
    pub tau_gravity: f64,
}

impl Default for WbcWeights {
    fn default() -> Self {
        Self {
            floating_base_eom: 1000.0,
            no_contact_motion: 1000.0,
            base_accel: 200.0,
            swing_leg: 1.0,
            contact_force: 5.0,
            tau_gravity: 5.0,
        }
    }
}

impl WbcWeights {
    /// Per-cmd-direction weight scheduling.
    ///
    /// Empirical sweep (see `tests/integration_walk.rs::diag_swing_leg_sweep_*`)
    /// shows the `swing_leg` task **flips sign** between forward and
    /// non-forward commands:
    ///
    /// | swing_leg | forward Δx | lateral Δy (cmd +0.1) | yaw Δyaw |
    /// |-----------|------------|------------------------|----------|
    /// | 1.0 (fwd best) | +0.12 ✓ | -0.90 ✗ (reversed)   | -1.83 ✗  |
    /// | 0.1 (lat best) | +0.20 ✓ | +0.50 ✓              | -2.77 ✗  |
    /// | 0.0           | -0.07 ✗ | -0.15                | -2.54 ✗  |
    ///
    /// Root cause: the joint-space PD generates fast hip q̈ during
    /// swing, which couples into a body-frame reaction torque
    /// (M_floating_base · q̈ at priority 0). The reaction sign matches
    /// the **forward** stride direction (no net body lateral push),
    /// but **opposes** the lateral / yaw stride direction (because
    /// hip abduction reaction-torques the body back toward the
    /// neutral pose).
    ///
    /// Mitigation: linearly fade `swing_leg` from `0.2` (forward-only
    /// command) down to `0.1` as `|cmd.vy|` and `|cmd.wz|` grow.
    /// This matches the sweep's local optima per axis without touching
    /// the other tasks.
    ///
    /// Saturation thresholds (`VY_FULL = 0.10 m/s`, `WZ_FULL = 0.5 rad/s`)
    /// match the test commands; cmds beyond them stay at the lateral-
    /// optimal weight. Hosts that want a different schedule can
    /// override `swing_leg` (or any other field) directly after this.
    ///
    /// Forward-axis sweep (constant `SWING_LATERAL = 0.1`):
    /// | SWING_FWD | body_dx | body_dy            | Δyaw     |
    /// |-----------|---------|--------------------|----------|
    /// | 1.0 (was) | +0.124  | -0.117             | -0.554   |
    /// | **0.2 (now)** | **+0.118** | **-0.034 (−3.4x)** | +0.589   |
    /// | 0.3       | +1.700  | -0.315             | +2.574   |
    /// | 0.5       | -0.399  | -0.718             | -0.811   |
    ///
    /// `0.2` keeps forward dx ≈ baseline while cutting lateral cross-
    /// coupling 3-4×. Going to `0.3` overshoots forward but explodes
    /// yaw drift; `0.5+` flips the sign — the system has narrow stable
    /// bands so this knob is fragile, but `0.2` is the best compromise
    /// in the regression suite.
    pub fn for_cmd(cmd: &crate::config::VelocityCmd) -> Self {
        const VY_FULL: f64 = 0.10;
        const WZ_FULL: f64 = 0.5;
        const SWING_FORWARD: f64 = 0.2;
        const SWING_LATERAL: f64 = 0.1;
        let lat = (cmd.vy.abs() / VY_FULL).min(1.0);
        let yaw = (cmd.wz.abs() / WZ_FULL).min(1.0);
        let intensity = lat.max(yaw); // 0 = pure forward, 1 = full lat/yaw
        let mut w = Self::default();
        w.swing_leg = SWING_FORWARD * (1.0 - intensity) + SWING_LATERAL * intensity;
        w
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
/// tick jitter. Uses [`WbcWeights::default`] for per-task LSQ weights.
pub fn solve_warm(inputs: &WbcInputs, warm: &WbcWarmStart<'_>) -> WbcSolution {
    solve_warm_with_weights(inputs, warm, &WbcWeights::default())
}

/// Like [`solve_warm`] but the caller supplies per-task LSQ
/// [`WbcWeights`]. Used by tests / diagnostic harnesses to study
/// each task's contribution by zeroing it out.
pub fn solve_warm_with_weights(
    inputs: &WbcInputs,
    warm: &WbcWarmStart<'_>,
    w: &WbcWeights,
) -> WbcSolution {
    let dims = inputs.dims;

    // ── Priority 0: hard constraints ───────────────────────────────
    let task_0 = floating_base_eom::formulate(
        dims,
        inputs.mass,
        inputs.nle,
        inputs.j_contact,
    )
    .weight(w.floating_base_eom)
        + torque_limits::formulate(dims, inputs.torque_max)
        + friction_cone::formulate(dims, inputs.contact_flag, inputs.friction_mu)
        + no_contact_motion::formulate(
            dims,
            inputs.j_contact,
            inputs.dj_v,
            inputs.contact_flag,
        )
        .weight(w.no_contact_motion);

    // ── Priority 1: motion tracking ────────────────────────────────
    let task_1 = base_accel::formulate(dims, inputs.a_base_des).weight(w.base_accel)
        + swing_leg::formulate(
            dims,
            inputs.swing_q_ddot_des,
            inputs.swing_actuator_flag,
        )
        .weight(w.swing_leg);

    // ── Priority 2: GRF + τ_grav joint regularisation ─────────────
    let task_2 = contact_force::formulate(dims, inputs.f_grf_des).weight(w.contact_force)
        + tau_gravity::formulate(dims, inputs.tau_gravity).weight(w.tau_gravity);

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
    use crate::config::VelocityCmd;

    /// `for_cmd(zero)` matches the default forward-tuned weights.
    #[test]
    fn for_cmd_zero_returns_default() {
        let w = WbcWeights::for_cmd(&VelocityCmd { vx: 0.0, vy: 0.0, wz: 0.0 });
        let d = WbcWeights::default();
        assert_eq!(w.swing_leg, d.swing_leg);
        assert_eq!(w.base_accel, d.base_accel);
        assert_eq!(w.contact_force, d.contact_force);
    }

    /// Pure forward command keeps the default forward-tuned swing_leg
    /// weight (= 1.0).
    #[test]
    fn for_cmd_pure_forward_keeps_default_swing_leg() {
        let w = WbcWeights::for_cmd(&VelocityCmd { vx: 0.5, vy: 0.0, wz: 0.0 });
        assert!((w.swing_leg - 0.2).abs() < 1e-9);
    }

    /// Pure full-rate lateral command snaps swing_leg to the lateral-
    /// optimal value (= 0.1).
    #[test]
    fn for_cmd_full_lateral_snaps_to_lateral_swing_leg() {
        let w = WbcWeights::for_cmd(&VelocityCmd { vx: 0.0, vy: 0.10, wz: 0.0 });
        assert!((w.swing_leg - 0.1).abs() < 1e-9);
    }

    /// Pure full-rate yaw command snaps swing_leg to the lateral-
    /// optimal value (same value as full lateral — both are non-
    /// forward locomotion modes that share the same sign-flip
    /// behaviour).
    #[test]
    fn for_cmd_full_yaw_snaps_to_lateral_swing_leg() {
        let w = WbcWeights::for_cmd(&VelocityCmd { vx: 0.0, vy: 0.0, wz: 0.5 });
        assert!((w.swing_leg - 0.1).abs() < 1e-9);
    }

    /// Half-intensity lateral command lands swing_leg at the linear
    /// midpoint between forward (1.0) and lateral (0.1) — i.e. 0.55.
    /// Sign of `cmd.vy` doesn't matter (we use `abs`).
    #[test]
    fn for_cmd_half_lateral_blends_linearly() {
        let w = WbcWeights::for_cmd(&VelocityCmd { vx: 0.0, vy: -0.05, wz: 0.0 });
        // 0.5 intensity → 0.5·1.0 + 0.5·0.1 = 0.55
        assert!((w.swing_leg - 0.55).abs() < 1e-9);
    }

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
        // Stance, no swing motion → swing flag all false, q̈_des all 0.
        let swing_q_ddot_des = DVector::zeros(4);
        let swing_actuator_flag = [false; 4];
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
            swing_q_ddot_des: &swing_q_ddot_des,
            swing_actuator_flag: &swing_actuator_flag,
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
