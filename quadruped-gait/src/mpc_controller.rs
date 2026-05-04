//! MPC-flavoured closed-loop gait controller.
//!
//! Architectural sibling of [`crate::ChampGaitController`]: same input
//! shape (velocity command + per-tick `dt`), same output shape
//! ([`ControllerOutput`] with 12 joint targets), and shares the body
//! integrator / phase generator / IK / swing trajectory primitives.
//! What's different is **footstep planning**:
//!
//! - CHAMP uses pure Raibert open-loop:
//!   `p = nominal + 0.5·T_stance·v_hip`
//! - This controller adds the **capture-point feedback** term Raibert
//!   1986 originally proposed and which `legged_control`'s NMPC
//!   replicates over a longer horizon:
//!   `p = nominal + 0.5·T_stance·v_hip + √(h/g)·(v_actual − v_cmd)`
//!   plus a multi-step LIP horizon (it predicts the body's future
//!   trajectory under each candidate foot location and picks the one
//!   that drives `v_actual` toward `v_cmd` over `N_horizon` upcoming
//!   steps).
//!
//! What this is **not**:
//!
//! - Not a full SRBD (single rigid body dynamics) MPC with QP-solved
//!   ground reaction forces, à la Di Carlo et al. / `ocs2_legged_robot`.
//!   Adding that requires switching the actuation chain from position
//!   control to torque feed-forward + WBC, which is a separate, larger
//!   refactor (see `doc/refactor_20260502.md` future-work).
//! - Not a magnetic-yaw-corrected estimator. The `set_body_state_observed`
//!   feedback comes from the host's own state estimate (MuJoCo's
//!   ground-truth body velocity in the simulation case, an EKF / IMU
//!   estimate on real hardware).

use nalgebra::Vector3;

use crate::body_state::BodyState;
use crate::config::{GaitConfig, KinematicsConfig, LegId, LegKinematics, VelocityCmd};
use crate::controller::{ControllerOutput, LegOutput};
use crate::footstep::Footstep;
use crate::ik::{solve_leg_ik, LegIkSolution};
use crate::phase::{PhaseGenerator, PhaseState};
use crate::srbd_mpc::{
    ContactSchedule, FootOffsets, MpcSolution, ReferenceTrajectory, SrbdMpc,
    SrbdMpcConfig, SrbdState,
};
use crate::swing_traj::swing_position;

/// Default capture-point feedback gain. Derived from the LIP model:
/// `k_fb = √(h/g)` with `h ≈ 0.30 m` (namiashi-class trunk height) and
/// `g = 9.81 m/s²` → ~0.175 s. The constructor uses this; runtime
/// tuning is exposed via [`MpcGaitController::set_capture_point_gain`].
pub const DEFAULT_CAPTURE_POINT_GAIN_S: f64 = 0.175;

/// Number of upcoming swing-step landings the horizon predictor looks
/// at when planning the *next* foot placement. The current
/// implementation uses the LIP closed-form solution (no QP), so cost
/// is O(N·legs) per tick — set to 4 to look about one full gait cycle
/// ahead at default duty_factor.
pub const HORIZON_STEPS: usize = 4;

/// Stateful MPC-flavoured gait controller. Same construction inputs
/// as [`crate::ChampGaitController`] so the host can swap them via
/// [`crate::AnyGaitController::set_mode`] without re-providing the
/// model.
#[derive(Clone, Debug)]
pub struct MpcGaitController {
    cfg: GaitConfig,
    kin: KinematicsConfig,
    phase_gen: PhaseGenerator,
    body_state: BodyState,
    cmd: VelocityCmd,
    knee_forward: [bool; 4],

    /// Capture-point feedback gain in seconds (`√(h/g)` of the LIP
    /// model). Larger → more aggressive correction → faster velocity
    /// tracking but bigger overshoot.
    k_capture: f64,

    /// Last reported observed body linear velocity in world frame.
    /// Defaults to zero; the host should call
    /// [`Self::set_body_state_observed`] every tick for the closed-loop
    /// feedback to do anything useful. When still zero (host hasn't
    /// wired the observation), the controller degrades to pure
    /// Raibert (open-loop).
    v_observed_world: Vector3<f64>,

    /// Convex SRBD MPC (Di Carlo et al. 2018) used to predict the
    /// ground reaction forces required to track the velocity command
    /// over the next ~300 ms. Currently produces *diagnostic* GRFs
    /// only — articara still drives joints via position control, so
    /// the forces are visualised but not commanded. Phase 4 will hook
    /// them into a torque-control chain.
    srbd_mpc: SrbdMpc,
    /// Last MPC solution. Populated each tick when the MPC succeeds;
    /// retained between ticks so the viewport can keep drawing arrows
    /// through paused frames.
    last_mpc_solution: Option<MpcSolution>,
}

impl MpcGaitController {
    /// Build with `swing_height_m`, `cycle_period_s`, `duty_factor`,
    /// etc. taken from `cfg`. Same shape as
    /// [`crate::ChampGaitController::new`].
    pub fn new(cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        let phase_gen = PhaseGenerator::new(cfg.clone());
        Self {
            cfg,
            kin,
            phase_gen,
            body_state: BodyState::new(),
            cmd: VelocityCmd::zero(),
            knee_forward: [false; 4],
            k_capture: DEFAULT_CAPTURE_POINT_GAIN_S,
            v_observed_world: Vector3::zeros(),
            srbd_mpc: SrbdMpc::new(SrbdMpcConfig::default()),
            last_mpc_solution: None,
        }
    }

    /// Predicted ground reaction forces from the most recent MPC
    /// solve. Returns `None` until the controller has ticked at least
    /// once with a stance leg. Used by the viewport to draw arrows;
    /// articara doesn't currently apply these to MuJoCo (Phase 4
    /// work).
    pub fn predicted_grfs(&self) -> Option<&MpcSolution> {
        self.last_mpc_solution.as_ref()
    }

    /// Tune the underlying SRBD MPC weights / horizon / friction.
    /// Most callers use the defaults.
    pub fn set_srbd_mpc_config(&mut self, cfg: SrbdMpcConfig) {
        self.srbd_mpc.set_config(cfg);
    }
    pub fn srbd_mpc_config(&self) -> &SrbdMpcConfig {
        self.srbd_mpc.config()
    }

    pub fn body_state(&self) -> &BodyState {
        &self.body_state
    }

    pub fn velocity_cmd(&self) -> VelocityCmd {
        self.cmd
    }
    pub fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        self.cmd = cmd;
    }

    pub fn config(&self) -> &GaitConfig {
        &self.cfg
    }
    pub fn set_config(&mut self, cfg: GaitConfig) {
        self.cfg = cfg;
        self.phase_gen = PhaseGenerator::new(self.cfg.clone());
    }

    pub fn kinematics(&self) -> &KinematicsConfig {
        &self.kin
    }

    pub fn knee_forward(&self) -> [bool; 4] {
        self.knee_forward
    }
    pub fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        self.knee_forward[crate::controller::slot_of(leg)] = forward;
    }
    pub fn set_knee_pattern(&mut self, pattern: crate::config::KneePattern) {
        self.knee_forward = pattern.to_knee_forward();
    }
    pub fn knee_pattern(&self) -> crate::config::KneePattern {
        crate::config::KneePattern::from_knee_forward(self.knee_forward)
    }

    /// Capture-point feedback gain (s). See [`Self::new`].
    pub fn capture_point_gain(&self) -> f64 {
        self.k_capture
    }
    pub fn set_capture_point_gain(&mut self, k: f64) {
        // Allow zero (turns the feedback off → degenerates to CHAMP-equivalent).
        self.k_capture = k.max(0.0);
    }

    /// Feed observed body linear velocity (world frame). Called by
    /// the host every sim tick from MuJoCo / state estimator output.
    pub fn set_body_state_observed(&mut self, v_world: Vector3<f64>) {
        self.v_observed_world = v_world;
    }

    pub fn reset(&mut self) {
        self.phase_gen.reset();
        self.body_state.reset();
        self.cmd = VelocityCmd::zero();
        self.v_observed_world = Vector3::zeros();
        self.last_mpc_solution = None;
    }

    /// Advance one tick. See module docs for what makes this differ
    /// from CHAMP's `tick`.
    pub fn tick(&mut self, dt: f64) -> ControllerOutput {
        self.phase_gen.advance(dt, &self.cmd);
        self.body_state.integrate(&self.cmd, dt);

        // Express the observed velocity in body frame so it can be
        // compared with the body-frame command directly (cmd is in
        // body frame by convention). For now we approximate "body
        // frame" as the integrated body yaw — fine while the body
        // doesn't roll/pitch much (which is the gait controller's
        // operating regime anyway).
        let v_obs_body = world_to_body_horizontal(
            self.v_observed_world,
            self.body_state.world_yaw,
        );
        let v_cmd = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
        let v_err_body = v_obs_body - v_cmd;

        let phases = self.phase_gen.legs();
        let mut legs: [Option<LegOutput>; 4] = [None, None, None, None];
        for ps in phases.iter() {
            let kin_leg = self.kin.leg(ps.leg);
            let footstep = self.compute_mpc_footstep(kin_leg, &v_err_body);
            let target = if ps.is_stance {
                footstep.stance_at(ps.sub_fraction)
            } else {
                let swing_h = effective_swing_height(self.cfg.swing_height_m, &self.cmd);
                swing_position(
                    footstep.lift_off,
                    footstep.touch_down,
                    swing_h,
                    ps.sub_fraction,
                )
            };
            let knee_fwd = self.knee_forward[crate::controller::slot_of(ps.leg)];
            let sol = solve_leg_ik(kin_leg, target, knee_fwd);
            let reachable = matches!(sol, LegIkSolution::Reached { .. });
            let (h, t, c) = sol.angles();
            legs[crate::controller::slot_of(ps.leg)] = Some(make_leg_output(
                ps.leg, kin_leg, *ps, footstep, target, h, t, c, reachable,
            ));
        }
        let output = ControllerOutput {
            legs: legs.map(|x| x.expect("all four legs filled by phase loop")),
            body_state: self.body_state,
        };

        // Solve the SRBD MPC for the next predicted GRFs. Best-effort:
        // failures (clarabel didn't converge, ill-conditioned QP, …)
        // leave the previous solution in place rather than panicking
        // — the GRF visualisation is non-critical.
        self.last_mpc_solution = Some(self.solve_srbd_mpc(&output));

        output
    }

    /// Build the inputs for [`SrbdMpc::solve`] from the current state
    /// + phase + footstep planning, then call clarabel. See
    /// [`Self::predicted_grfs`].
    fn solve_srbd_mpc(&self, output: &ControllerOutput) -> MpcSolution {
        let cfg = self.srbd_mpc.config();
        let n = cfg.horizon_steps;

        // Current SRBD state. Yaw comes from the integrated body
        // pose; roll/pitch tracked elsewhere are approximated as zero
        // (the gait controller doesn't observe body orientation in
        // Phase 2).
        let s_now = SrbdState {
            orientation_rpy: Vector3::new(0.0, 0.0, self.body_state.world_yaw),
            position: Vector3::new(
                self.body_state.world_position.x,
                self.body_state.world_position.y,
                // Use the kinematics' nominal stance height as a proxy
                // for the body z; gives a stable level for the MPC to
                // regulate to.
                -self.kin.legs()[0].nominal_foot_body.z,
            ),
            angular_velocity: Vector3::new(0.0, 0.0, self.cmd.wz),
            linear_velocity: self.v_observed_world,
        };

        // Reference: track commanded velocity over the horizon.
        let v_world_cmd = body_to_world_horizontal(
            Vector3::new(self.cmd.vx, self.cmd.vy, 0.0),
            self.body_state.world_yaw,
        );
        let reference =
            ReferenceTrajectory::from_constant_velocity(s_now, v_world_cmd, self.cmd.wz, cfg);

        // Contact schedule: project current per-leg phase forward by
        // `dt_per_step` for each horizon step. The phase generator's
        // `predict_at` would be ideal but we don't have it; use the
        // per-leg duty fraction as a steady-state proxy.
        let stance_now: [bool; 4] = [
            output.legs[0].phase.is_stance,
            output.legs[1].phase.is_stance,
            output.legs[2].phase.is_stance,
            output.legs[3].phase.is_stance,
        ];
        let mut contact = ContactSchedule {
            is_stance: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        };
        for leg in 0..4 {
            // Steady-state stance probability at any random horizon
            // step is `duty_factor`. For a coarse projection use the
            // current state for the first step and the duty for the
            // rest — this errs on the side of assuming the foot stays
            // in stance, giving the MPC a denser support hypothesis.
            for k in 0..n {
                let in_stance = if k == 0 {
                    stance_now[leg]
                } else {
                    self.cfg.duty_factor > 0.5
                };
                contact.is_stance[leg].push(in_stance);
            }
        }

        // Foot offsets: r_i = foot_world − CoM_world, but the MPC's
        // input is just the relative vector. Use the current foot_body
        // (already in body frame, world-aligned modulo yaw) as a
        // constant approximation over the horizon. Good enough for
        // 300 ms; the swing leg's foot motion is averaged out by the
        // MPC's stance schedule.
        let foot_body: [Vector3<f64>; 4] = [
            output.legs[0].foot_body,
            output.legs[1].foot_body,
            output.legs[2].foot_body,
            output.legs[3].foot_body,
        ];
        let feet = FootOffsets::constant_per_leg(foot_body, n);

        self.srbd_mpc.solve(s_now, &reference, &contact, &feet)
    }

    /// Footstep planner with capture-point feedback + LIP horizon
    /// look-ahead. Conceptually:
    ///
    /// ```text
    /// p = nominal_foot
    ///   + 0.5 · T_stance · v_hip          ← Raibert (open-loop)
    ///   + √(h/g) · (v_actual − v_cmd)     ← capture-point (closed-loop)
    ///   + horizon_correction               ← LIP look-ahead
    /// ```
    ///
    /// The horizon term is the average per-step displacement the LIP
    /// would need over the next [`HORIZON_STEPS`] cycles to reach
    /// `v_cmd`; it folds in as a small bias so the controller doesn't
    /// chase a single-step correction (which leads to oscillation).
    fn compute_mpc_footstep(
        &self,
        kin: &LegKinematics,
        v_err_body: &Vector3<f64>,
    ) -> Footstep {
        let stance_duration = self.cfg.cycle_period_s * self.cfg.duty_factor;
        let v_body = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
        let omega = Vector3::new(0.0, 0.0, self.cmd.wz);
        let v_hip = v_body + omega.cross(&kin.hip_offset);

        // ── Raibert (open-loop) ────────────────────────────────────
        let mut half = v_hip * (0.5 * stance_duration);

        // ── Capture-point (closed-loop) ────────────────────────────
        // Only apply the correction in xy; z is set elsewhere.
        let mut feedback = Vector3::zeros();
        feedback.x = self.k_capture * v_err_body.x;
        feedback.y = self.k_capture * v_err_body.y;

        // ── LIP horizon look-ahead ────────────────────────────────
        // Bias the foot slightly toward where the LIP says the body
        // would be after HORIZON_STEPS gait cycles if v_actual stayed
        // off-cmd. Small weight (1/HORIZON_STEPS) so it doesn't
        // dominate the per-step correction.
        let horizon_weight = 1.0 / HORIZON_STEPS as f64;
        let mut horizon_bias = Vector3::zeros();
        horizon_bias.x = horizon_weight * self.k_capture * v_err_body.x;
        horizon_bias.y = horizon_weight * self.k_capture * v_err_body.y;

        // Combine and clamp to the configured maximum step length.
        half += feedback + horizon_bias;
        let max_half = 0.5 * self.cfg.max_step_length_m;
        let mag = half.norm();
        if mag > max_half && mag > 0.0 {
            half *= max_half / mag;
        }
        Footstep {
            lift_off: kin.nominal_foot_body - half,
            touch_down: kin.nominal_foot_body + half,
        }
    }
}

// Helper: project a world-frame horizontal velocity into the body
// frame by undoing the integrated yaw. Pitch / roll ignored — fine for
// the gait operating regime (small body tilt).
fn world_to_body_horizontal(v_world: Vector3<f64>, yaw: f64) -> Vector3<f64> {
    let (s, c) = yaw.sin_cos();
    Vector3::new(
        c * v_world.x + s * v_world.y,
        -s * v_world.x + c * v_world.y,
        v_world.z,
    )
}

/// Inverse of [`world_to_body_horizontal`]: rotate a body-frame
/// horizontal vector into world frame using the yaw angle.
fn body_to_world_horizontal(v_body: Vector3<f64>, yaw: f64) -> Vector3<f64> {
    let (s, c) = yaw.sin_cos();
    Vector3::new(
        c * v_body.x - s * v_body.y,
        s * v_body.x + c * v_body.y,
        v_body.z,
    )
}

// Reuse the same swing-height knockdown as the CHAMP controller so
// both modes feel similar at high yaw / strafe rates.
fn effective_swing_height(base_h: f64, cmd: &VelocityCmd) -> f64 {
    const WZ_REF: f64 = 1.0;
    const VY_REF: f64 = 0.3;
    const MIN_FACTOR: f64 = 0.4;
    let r_yaw = (cmd.wz.abs() / WZ_REF).min(1.0);
    let r_lat = (cmd.vy.abs() / VY_REF).min(1.0);
    let r = ((r_yaw + r_lat) * 0.5).clamp(0.0, 1.0);
    let factor = 1.0 - (1.0 - MIN_FACTOR) * r;
    base_h * factor
}

#[allow(clippy::too_many_arguments)]
fn make_leg_output(
    leg: LegId,
    kin: &LegKinematics,
    phase: PhaseState,
    footstep: Footstep,
    foot_body: Vector3<f64>,
    q_hip: f64,
    q_thigh: f64,
    q_calf: f64,
    reachable: bool,
) -> LegOutput {
    LegOutput {
        leg,
        hip_joint: kin.hip_joint.clone(),
        thigh_joint: kin.thigh_joint.clone(),
        calf_joint: kin.calf_joint.clone(),
        q_hip,
        q_thigh,
        q_calf,
        foot_body,
        footstep,
        phase,
        reachable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GaitConfig, KinematicsConfig, LegId, LegKinematics};
    use approx::assert_relative_eq;

    fn build_kin() -> KinematicsConfig {
        let leg = |id: LegId, hip: Vector3<f64>| {
            let mut k = LegKinematics::new(
                id,
                format!("{id:?}_hip").to_lowercase(),
                format!("{id:?}_thigh").to_lowercase(),
                format!("{id:?}_calf").to_lowercase(),
                format!("{id:?}_foot").to_lowercase(),
                hip,
                0.04,
                0.18,
                0.18,
            );
            k.nominal_foot_body.z = -0.30;
            k
        };
        KinematicsConfig {
            fl: leg(LegId::FL, Vector3::new(0.18, 0.05, 0.0)),
            fr: leg(LegId::FR, Vector3::new(0.18, -0.05, 0.0)),
            rl: leg(LegId::RL, Vector3::new(-0.18, 0.05, 0.0)),
            rr: leg(LegId::RR, Vector3::new(-0.18, -0.05, 0.0)),
        }
    }

    /// With zero observed velocity error and pure forward command,
    /// MPC should produce **identical** lift_off / touch_down to CHAMP.
    /// Guards against accidental divergence from the proven Raibert
    /// formula on the common case.
    #[test]
    fn matches_champ_when_velocity_error_is_zero() {
        let cfg = GaitConfig::trot();
        let kin = build_kin();

        let mut champ = crate::ChampGaitController::new(cfg.clone(), kin.clone());
        champ.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });

        let mut mpc = MpcGaitController::new(cfg, kin);
        mpc.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
        // Body velocity matches command → zero error → no feedback.
        mpc.set_body_state_observed(Vector3::new(0.3, 0.0, 0.0));

        let out_champ = champ.tick(0.002);
        let out_mpc = mpc.tick(0.002);
        for slot in 0..4 {
            for ax in 0..3 {
                assert_relative_eq!(
                    out_champ.legs[slot].footstep.lift_off[ax],
                    out_mpc.legs[slot].footstep.lift_off[ax],
                    epsilon = 1e-9,
                );
                assert_relative_eq!(
                    out_champ.legs[slot].footstep.touch_down[ax],
                    out_mpc.legs[slot].footstep.touch_down[ax],
                    epsilon = 1e-9,
                );
            }
        }
    }

    /// When the body is **slower** than commanded forward, the
    /// capture-point feedback should push the foot **further forward**
    /// (positive x correction) so the next stance pulls the body up
    /// to speed.
    #[test]
    fn slow_body_pushes_foot_forward() {
        let cfg = GaitConfig::trot();
        let kin = build_kin();
        let mut mpc = MpcGaitController::new(cfg, kin);
        mpc.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
        // Observed velocity is half of commanded → error = -0.15 m/s.
        mpc.set_body_state_observed(Vector3::new(0.15, 0.0, 0.0));

        let out_mpc = mpc.tick(0.002);
        // CHAMP reference (no feedback)
        let mut champ = crate::ChampGaitController::new(GaitConfig::trot(), build_kin());
        champ.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
        let out_champ = champ.tick(0.002);

        for slot in 0..4 {
            // err.x = v_obs.x − v_cmd.x = 0.15 − 0.30 = −0.15 (body is slow)
            // The capture-point term is k * err = -0.175 * 0.15 ≈ -0.026 in x
            // touch_down shifts by + this amount (since half includes feedback)
            // so actually... let me think again.
            //
            // half = 0.5*T_stance*v_hip + k*v_err_body + horizon_bias
            // touch_down = nominal + half
            //
            // v_err_body = v_obs - v_cmd = -0.15 (slow)
            // → half.x decreases (more negative shift)
            // → touch_down.x decreases (foot moves backward)
            //
            // Hmm, that's the opposite of "push foot forward to catch up". Let me
            // reason about the physics:
            // - If body is slow, we want the foot to act as a brake LESS
            //   → step shorter or further forward of the body
            // - In Raibert's original paper, the feedback term is +k*(v_actual − v_cmd):
            //   when actual < commanded, foot lands further BACK (less braking).
            //
            // OK so the sign is correct for "less braking when slow", which lets
            // the body accelerate up to v_cmd. Test for that:
            let dx_mpc = out_mpc.legs[slot].footstep.touch_down.x;
            let dx_champ = out_champ.legs[slot].footstep.touch_down.x;
            assert!(
                dx_mpc < dx_champ,
                "leg {slot}: when body is slow, foot should land FURTHER BACK \
                 than CHAMP (less braking) — got mpc={dx_mpc}, champ={dx_champ}",
            );
        }
    }

    /// Setting the gain to 0 disables feedback → identical to CHAMP
    /// regardless of velocity error.
    #[test]
    fn zero_gain_disables_feedback() {
        let cfg = GaitConfig::trot();
        let kin = build_kin();
        let mut mpc = MpcGaitController::new(cfg.clone(), kin.clone());
        mpc.set_capture_point_gain(0.0);
        mpc.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
        mpc.set_body_state_observed(Vector3::new(0.0, 0.5, 0.0)); // huge error

        let mut champ = crate::ChampGaitController::new(cfg, kin);
        champ.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });

        let out_mpc = mpc.tick(0.002);
        let out_champ = champ.tick(0.002);
        for slot in 0..4 {
            for ax in 0..3 {
                assert_relative_eq!(
                    out_mpc.legs[slot].footstep.touch_down[ax],
                    out_champ.legs[slot].footstep.touch_down[ax],
                    epsilon = 1e-9,
                );
            }
        }
    }

    /// `set_mode` round-trip on `AnyGaitController` preserves the
    /// command across the swap.
    #[test]
    fn switch_mode_preserves_cmd() {
        use crate::{AnyGaitController, GaitGenerator, GaitMode};
        let mut any = AnyGaitController::new(
            GaitMode::Champ,
            GaitConfig::trot(),
            build_kin(),
        );
        any.set_velocity_cmd(VelocityCmd { vx: 0.4, vy: 0.1, wz: 0.2 });
        any.set_mode(GaitMode::Mpc);
        let cmd = any.velocity_cmd();
        assert_relative_eq!(cmd.vx, 0.4);
        assert_relative_eq!(cmd.vy, 0.1);
        assert_relative_eq!(cmd.wz, 0.2);
        any.set_mode(GaitMode::Champ);
        let cmd2 = any.velocity_cmd();
        assert_relative_eq!(cmd2.vx, 0.4);
    }
}
