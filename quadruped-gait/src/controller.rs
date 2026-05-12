//! Top-level gait controller.
//!
//! Glues together [`PhaseGenerator`], [`compute_footstep`], the swing /
//! stance trajectories, and per-leg analytical IK. Each call to
//! [`GaitController::tick`] advances the gait by `dt` seconds and returns
//! a [`ControllerOutput`] holding the joint targets, foot positions, and
//! diagnostic info for the four legs.
//!
//! # Usage
//!
//! ```no_run
//! use quadruped_gait::*;
//! # let kin: KinematicsConfig = unimplemented!();
//! let cfg = GaitConfig::trot();
//! let mut ctrl = GaitController::new(cfg, kin);
//! ctrl.set_velocity_cmd(VelocityCmd { vx: 0.3, vy: 0.0, wz: 0.0 });
//! loop {
//!     let out = ctrl.tick(0.002);
//!     for (name, q) in out.iter_joint_targets() {
//!         // hand off to the simulator's set_position_target
//!     }
//! }
//! ```
//!
//! The controller is body-frame open-loop: foot targets are computed in
//! the current body frame and the hosts feeds the same `VelocityCmd` until
//! it changes. There's no IMU/encoder feedback yet — that's a v0.2 task.

use nalgebra::Vector3;

use crate::body_state::BodyState;
use crate::config::{GaitConfig, KinematicsConfig, LegId, LegKinematics, VelocityCmd};
use crate::footstep::{compute_footstep, Footstep};
use crate::ik::{solve_leg_ik, LegIkSolution};
use crate::phase::{PhaseGenerator, PhaseState};
use crate::swing_traj::swing_position;

/// Per-leg controller output. Joint names are duplicated here (rather
/// than only kept in [`KinematicsConfig`]) so consumers iterating over
/// the targets don't need to keep both structures.
#[derive(Clone, Debug)]
pub struct LegOutput {
    pub leg: LegId,
    pub hip_joint: String,
    pub thigh_joint: String,
    pub calf_joint: String,
    pub q_hip: f64,
    pub q_thigh: f64,
    pub q_calf: f64,
    pub foot_body: Vector3<f64>,
    pub footstep: Footstep,
    pub phase: PhaseState,
    /// `false` when the foot target was outside the leg workspace and the
    /// IK had to clamp. Hosts should warn in this case (the angles still
    /// produce a valid posture, just not the requested foot pose).
    pub reachable: bool,
}

/// Aggregate output of one [`GaitController::tick`]. Per-leg slots are in
/// canonical [`LegId::ALL`] order (FL, FR, RL, RR).
#[derive(Clone, Debug)]
pub struct ControllerOutput {
    pub legs: [LegOutput; 4],
    /// Convenience snapshot of the integrated body pose at the end of
    /// this tick. Updated even when `cmd.is_zero()`.
    pub body_state: BodyState,
}

impl ControllerOutput {
    /// Iterate over all 12 (joint_name, target_q) pairs. The order is
    /// FL{hip,thigh,calf}, FR{...}, RL{...}, RR{...}.
    pub fn iter_joint_targets(&self) -> impl Iterator<Item = (&str, f64)> {
        self.legs.iter().flat_map(|l| {
            [
                (l.hip_joint.as_str(), l.q_hip),
                (l.thigh_joint.as_str(), l.q_thigh),
                (l.calf_joint.as_str(), l.q_calf),
            ]
        })
    }

    /// Return the leg output for `leg`. Constant-time lookup since the
    /// canonical layout is fixed.
    pub fn leg(&self, leg: LegId) -> &LegOutput {
        &self.legs[slot_of(leg)]
    }

    /// True only if all four legs reported a reachable IK solution.
    pub fn all_reachable(&self) -> bool {
        self.legs.iter().all(|l| l.reachable)
    }
}

/// Stateful gait controller. One instance per robot.
#[derive(Clone, Debug)]
pub struct GaitController {
    cfg: GaitConfig,
    kin: KinematicsConfig,
    phase_gen: PhaseGenerator,
    body_state: BodyState,
    cmd: VelocityCmd,
    /// Per-leg knee direction. `true` = knee bends forward (front-leg
    /// style), `false` = knee bends backward (rear-leg style). Default is
    /// all-false because that's the IK's "natural" sign convention; flip
    /// fronts to `true` after construction if your URDF requires it.
    knee_forward: [bool; 4],
}

/// Reduce the effective swing height when the command contains
/// significant yaw / lateral content. Per-leg stride amplitudes
/// become asymmetric in those modes (outer-radius leg sweeps further
/// than inner) and a tall swing amplifies the asymmetric body
/// reaction at touchdown / lift-off. A multiplicative knockdown is
/// blunt but cheap and predictable; the user tunes the *base*
/// `swing_height_m` and this function preserves that baseline for
/// pure forward motion (the common case).
///
/// Mapping (by `r = (|wz|/wz_ref + |vy|/vy_ref) / 2`, clamped 0..1):
/// - `r = 0` → 1.0 × base height (pure forward / stationary)
/// - `r = 1` → 0.4 × base height (full turn or full strafe)
/// - linear interpolation in between
///
/// `wz_ref = 1.0 rad/s` and `vy_ref = 0.3 m/s` are reasonable
/// "moderate" thresholds for namiashi-class quadrupeds; constants
/// here rather than gait config so a future refactor can promote
/// them to [`crate::config::GaitConfig`] if real robots disagree.
fn effective_swing_height(base_h: f64, cmd: &VelocityCmd) -> f64 {
    const WZ_REF: f64 = 1.0;     // rad/s
    const VY_REF: f64 = 0.3;     // m/s
    const MIN_FACTOR: f64 = 0.4; // never drop swing below 40% of base

    let r_yaw = (cmd.wz.abs() / WZ_REF).min(1.0);
    let r_lat = (cmd.vy.abs() / VY_REF).min(1.0);
    // Combine — average so a moderate yaw + moderate strafe doesn't
    // cancel the reduction. Clamp 0..1.
    let r = ((r_yaw + r_lat) * 0.5).clamp(0.0, 1.0);
    let factor = 1.0 - (1.0 - MIN_FACTOR) * r;
    base_h * factor
}

pub(crate) fn slot_of(id: LegId) -> usize {
    match id {
        LegId::FL => 0,
        LegId::FR => 1,
        LegId::RL => 2,
        LegId::RR => 3,
    }
}

impl GaitController {
    pub fn new(cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        Self {
            phase_gen: PhaseGenerator::new(cfg.clone()),
            cfg,
            kin,
            body_state: BodyState::new(),
            cmd: VelocityCmd::zero(),
            knee_forward: [false; 4],
        }
    }

    pub fn config(&self) -> &GaitConfig {
        &self.cfg
    }
    pub fn kinematics(&self) -> &KinematicsConfig {
        &self.kin
    }
    pub fn velocity_cmd(&self) -> VelocityCmd {
        self.cmd
    }
    pub fn body_state(&self) -> BodyState {
        self.body_state
    }

    pub fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        self.cmd = cmd;
    }

    /// Replace the gait config. The phase generator picks up the new
    /// timing on its next [`PhaseGenerator::advance`] call; current cycle
    /// phase is preserved so a live transition between gaits doesn't
    /// snap back to t=0.
    pub fn set_config(&mut self, cfg: GaitConfig) {
        self.phase_gen.set_config(cfg.clone());
        self.cfg = cfg;
    }

    /// Replace the kinematics. Useful when a model is swapped at runtime.
    pub fn set_kinematics(&mut self, kin: KinematicsConfig) {
        self.kin = kin;
    }

    /// Set the knee direction for one leg. See the [`Self::knee_forward`]
    /// docs (default `false` for all legs).
    pub fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        self.knee_forward[slot_of(leg)] = forward;
    }

    /// Apply a symmetric front/rear knee pattern (the four-pattern shorthand
    /// `<<`, `<>`, `><`, `>>` — see [`crate::config::KneePattern`]). Sets
    /// the per-leg knee_forward flags in one call.
    pub fn set_knee_pattern(&mut self, pattern: crate::config::KneePattern) {
        self.knee_forward = pattern.to_knee_forward();
    }

    /// Read back the current knee configuration as a [`KneePattern`].
    /// When the per-leg flags are asymmetric (set via `set_knee_forward`
    /// rather than `set_knee_pattern`) the result is best-effort — see
    /// [`KneePattern::from_knee_forward`].
    pub fn knee_pattern(&self) -> crate::config::KneePattern {
        crate::config::KneePattern::from_knee_forward(self.knee_forward)
    }

    /// Read-only access to the per-leg knee_forward array, indexed
    /// `[FL, FR, RL, RR]`.
    pub fn knee_forward(&self) -> [bool; 4] {
        self.knee_forward
    }

    /// Reset phase + body integrator to the cycle origin and zero command.
    pub fn reset(&mut self) {
        self.phase_gen.reset();
        self.body_state.reset();
        self.cmd = VelocityCmd::zero();
    }

    /// Advance the gait by `dt` seconds. Returns the per-leg outputs.
    pub fn tick(&mut self, dt: f64) -> ControllerOutput {
        // 1. Advance the global phase (frozen when cmd is zero).
        self.phase_gen.advance(dt, &self.cmd);
        // 2. Integrate the body pose for diagnostics.
        self.body_state.integrate(&self.cmd, dt);
        // 3. For each leg, decide stance vs swing, compute the target
        //    foot position via Raibert footstep + Bezier swing curve,
        //    and run the IK.
        let phases = self.phase_gen.legs();
        // Effective swing height for this tick. Reduce when the
        // command has significant turn / lateral content — those modes
        // make the per-leg stride amplitudes asymmetric, so a tall
        // swing combined with the asymmetric loading rocks the trunk
        // visibly even with the C¹-continuous swing curve.
        //
        // Scaling factor: 1.0 for pure forward, dropping toward
        // `min_factor` (40%) when |wz| or |vy| dominates. This is a
        // pragmatic dampening rather than a derived optimum — tuned so
        // namiashi turns smoothly without dragging the foot.
        let swing_h = effective_swing_height(self.cfg.swing_height_m, &self.cmd);
        // Pre-compute leg outputs in canonical order, then assemble.
        let mut legs: [Option<LegOutput>; 4] = [None, None, None, None];
        for ps in phases.iter() {
            let kin_leg = self.kin.leg(ps.leg);
            let footstep = compute_footstep(kin_leg, &self.cfg, &self.cmd);
            let target = if ps.is_stance {
                footstep.stance_at(ps.sub_fraction)
            } else {
                swing_position(
                    footstep.lift_off,
                    footstep.touch_down,
                    swing_h,
                    ps.sub_fraction,
                )
            };
            let knee_fwd = self.knee_forward[slot_of(ps.leg)];
            let sol = solve_leg_ik(kin_leg, target, knee_fwd);
            let reachable = matches!(sol, LegIkSolution::Reached { .. });
            let (h, t, c) = sol.angles();
            legs[slot_of(ps.leg)] = Some(per_leg_output(
                ps.leg, kin_leg, *ps, footstep, target, h, t, c, reachable,
            ));
        }
        ControllerOutput {
            legs: legs.map(|x| x.expect("all four legs filled by phase loop")),
            body_state: self.body_state,
        }
    }
}

fn per_leg_output(
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
    use approx::assert_relative_eq;

    fn build_kin() -> KinematicsConfig {
        let mk = |leg: LegId, sx: f64, sy: f64, prefix: &str| {
            let mut k = LegKinematics::new(
                leg,
                format!("{prefix}_hip"),
                format!("{prefix}_thigh"),
                format!("{prefix}_calf"),
                format!("{prefix}_foot"),
                Vector3::new(sx * 0.18, sy * 0.05, 0.0),
                0.04,
                0.18,
                0.18,
            );
            // Bent-knee neutral so swings have headroom.
            k.nominal_foot_body.z = k.hip_offset.z - 0.36 * 0.92;
            k
        };
        KinematicsConfig {
            fl: mk(LegId::FL, 1.0, 1.0, "FL"),
            fr: mk(LegId::FR, 1.0, -1.0, "FR"),
            rl: mk(LegId::RL, -1.0, 1.0, "RL"),
            rr: mk(LegId::RR, -1.0, -1.0, "RR"),
        }
    }

    #[test]
    fn iter_joint_targets_returns_twelve_entries_in_canonical_order() {
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        ctrl.set_velocity_cmd(VelocityCmd { vx: 0.2, ..Default::default() });
        let out = ctrl.tick(0.002);
        let names: Vec<&str> = out.iter_joint_targets().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 12);
        assert_eq!(names[0], "FL_hip");
        assert_eq!(names[1], "FL_thigh");
        assert_eq!(names[2], "FL_calf");
        assert_eq!(names[3], "FR_hip");
        // ...
        assert_eq!(names[11], "RR_calf");
    }

    #[test]
    fn zero_cmd_holds_nominal_pose_constant() {
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        // Don't call set_velocity_cmd; default is zero.
        let out0 = ctrl.tick(0.002);
        for _ in 0..200 {
            let out = ctrl.tick(0.002);
            for slot in 0..4 {
                assert_relative_eq!(
                    out.legs[slot].q_hip, out0.legs[slot].q_hip,
                    epsilon = 1e-12,
                );
                assert_relative_eq!(
                    out.legs[slot].q_thigh, out0.legs[slot].q_thigh,
                    epsilon = 1e-12,
                );
                assert_relative_eq!(
                    out.legs[slot].q_calf, out0.legs[slot].q_calf,
                    epsilon = 1e-12,
                );
                assert!(out.legs[slot].phase.is_stance);
            }
        }
        // Body integrator stays at origin.
        assert_relative_eq!(ctrl.body_state().world_position.x, 0.0);
    }

    #[test]
    fn reset_returns_to_t_zero_state() {
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        ctrl.set_velocity_cmd(VelocityCmd { vx: 0.5, ..Default::default() });
        // Walk for half a cycle.
        for _ in 0..100 {
            ctrl.tick(0.002);
        }
        ctrl.reset();
        assert_eq!(ctrl.velocity_cmd(), VelocityCmd::zero());
        assert_eq!(ctrl.body_state().world_position, Vector3::zeros());
        // After reset, every leg should be in stance with sub_fraction 0.
        let out = ctrl.tick(0.002);
        for slot in 0..4 {
            assert!(out.legs[slot].phase.is_stance);
        }
    }

    #[test]
    fn stance_foot_moves_backward_in_body_frame_when_walking_forward() {
        // Regression test for a Phase 2 bug where Footstep::stance_at
        // interpolated lift_off → touch_down (back to front) instead of
        // touch_down → lift_off (front to back). When a quadruped walks
        // forward (+vx), the foot in stance must sweep BACKWARD relative
        // to the body — that's what creates the forward-pushing reaction
        // force at contact. The bug had foot moving forward, propelling
        // the robot the wrong way.
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        ctrl.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });

        let dt = 0.002;
        let mut samples: Vec<(f64, bool, f64)> = Vec::new(); // (cycle pos, is_stance, foot.x)
        let n = (ctrl.config().cycle_period_s / dt) as usize + 1;
        for _ in 0..n {
            let out = ctrl.tick(dt);
            let fl = &out.legs[0]; // FL slot
            samples.push((fl.phase.cycle_position, fl.phase.is_stance, fl.foot_body.x));
        }

        // Find a stance run and verify foot.x decreases monotonically.
        let stance_runs: Vec<&(f64, bool, f64)> =
            samples.iter().filter(|s| s.1).collect();
        assert!(
            stance_runs.len() >= 4,
            "expected several stance samples, got {}",
            stance_runs.len(),
        );
        let first_x = stance_runs.first().unwrap().2;
        let last_x = stance_runs.last().unwrap().2;
        assert!(
            last_x < first_x,
            "stance foot.x must decrease (move back in body frame) when walking +x; \
             got first={first_x} last={last_x}",
        );
    }

    #[test]
    fn foot_trajectory_independent_of_knee_pattern() {
        // The body-frame foot trajectory (and therefore the world-frame
        // body motion) must be the same for `<<` and `>>` patterns — they
        // pick different IK branches that resolve to the same foot point.
        // If a future change accidentally couples the trajectory to the
        // knee_forward flag, this test catches the regression that would
        // otherwise show up as "<< walks forward, >> walks backward."
        use crate::config::KneePattern;

        let dt = 0.002;
        let n = 200; // half a default trot cycle

        let mut snapshots: Vec<Vec<f64>> = Vec::new(); // per-pattern foot.x sequence
        for pattern in [KneePattern::BothBack, KneePattern::BothForward] {
            let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
            ctrl.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
            ctrl.set_knee_pattern(pattern);
            let mut xs = Vec::with_capacity(n);
            for _ in 0..n {
                let out = ctrl.tick(dt);
                xs.push(out.legs[0].foot_body.x);
            }
            snapshots.push(xs);
        }

        // The two trajectories must agree to better than 1e-9 since the
        // foot target is computed identically; only the joint-space IK
        // branch differs.
        for i in 0..n {
            assert_relative_eq!(
                snapshots[0][i],
                snapshots[1][i],
                epsilon = 1e-9,
            );
        }
    }

    #[test]
    fn knee_pattern_round_trip_through_controller() {
        use crate::config::KneePattern;
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        for pattern in KneePattern::ALL {
            ctrl.set_knee_pattern(pattern);
            assert_eq!(ctrl.knee_pattern(), pattern);
            assert_eq!(ctrl.knee_forward(), pattern.to_knee_forward());
        }
    }

    #[test]
    fn forward_walk_advances_body_position() {
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        ctrl.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
        for _ in 0..500 {
            ctrl.tick(0.002); // 1 second total
        }
        // 0.3 m/s for 1s → ~0.3 m
        assert_relative_eq!(ctrl.body_state().world_position.x, 0.3, epsilon = 1e-9);
    }

    /// `effective_swing_height` preserves the base value for pure
    /// forward / stationary commands (the common case) so users who
    /// haven't tuned for turn / strafe see the same gait as before.
    #[test]
    fn effective_swing_height_unchanged_for_forward_motion() {
        let base = 0.04;
        let h = effective_swing_height(
            base,
            &VelocityCmd { vx: 0.5, vy: 0.0, wz: 0.0 },
        );
        assert_relative_eq!(h, base);
        let h_stand =
            effective_swing_height(base, &VelocityCmd { vx: 0.0, vy: 0.0, wz: 0.0 });
        assert_relative_eq!(h_stand, base);
    }

    /// Strong yaw → swing height reduced toward the floor (40% min).
    #[test]
    fn effective_swing_height_reduced_under_yaw() {
        let base = 0.04;
        let h_full_yaw = effective_swing_height(
            base,
            &VelocityCmd { vx: 0.0, vy: 0.0, wz: 1.0 },
        );
        // r = (1/1 + 0/0.3) / 2 = 0.5 → factor = 1 - 0.6·0.5 = 0.7
        assert_relative_eq!(h_full_yaw, base * 0.7, epsilon = 1e-9);

        // Combined full yaw + full strafe → r = 1 → factor = 0.4
        let h_extreme = effective_swing_height(
            base,
            &VelocityCmd { vx: 0.0, vy: 0.5, wz: 2.0 },
        );
        assert_relative_eq!(h_extreme, base * 0.4, epsilon = 1e-9);
    }

    /// Strong lateral motion alone → swing height reduced.
    #[test]
    fn effective_swing_height_reduced_under_strafe() {
        let base = 0.04;
        let h = effective_swing_height(
            base,
            &VelocityCmd { vx: 0.0, vy: 0.3, wz: 0.0 },
        );
        // r = (0 + 0.3/0.3) / 2 = 0.5 → factor = 0.7
        assert_relative_eq!(h, base * 0.7, epsilon = 1e-9);
    }

    /// Reduction never drops the swing below the 40% floor regardless
    /// of how aggressive the command is. Below ~30% the foot starts to
    /// scrape the ground.
    #[test]
    fn effective_swing_height_floor_at_40pct() {
        let base = 0.04;
        let h = effective_swing_height(
            base,
            &VelocityCmd { vx: 0.0, vy: 100.0, wz: 100.0 }, // absurd
        );
        assert!(h >= base * 0.4 - 1e-9);
        assert!(h <= base * 0.4 + 1e-9);
    }

    /// March-in-place semantics: a tiny but nonzero command should
    /// keep the phase generator cycling (it freezes at `cmd.is_zero()`)
    /// while the Raibert stride amplitude collapses to ~0, so feet lift
    /// visibly in swing but the body doesn't translate. The GUI panel's
    /// `👣` button relies on this — see `src/app/gait_panel.rs`.
    #[test]
    fn tiny_cmd_lifts_swing_legs_without_horizontal_motion() {
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        ctrl.set_velocity_cmd(VelocityCmd { vx: 1e-6, vy: 0.0, wz: 0.0 });
        let mut max_z = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_x_excursion = 0.0_f64;
        // Full cycle ≈ 0.4 s at default trot params → tick 250× at 2 ms
        // to comfortably cover one cycle.
        for _ in 0..250 {
            let out = ctrl.tick(0.002);
            for slot in 0..4 {
                let nom_x = ctrl.kinematics().legs()[slot].nominal_foot_body.x;
                let z = out.legs[slot].foot_body.z;
                max_z = max_z.max(z);
                min_z = min_z.min(z);
                max_x_excursion = max_x_excursion.max(
                    (out.legs[slot].foot_body.x - nom_x).abs(),
                );
            }
        }
        let z_span = max_z - min_z;
        // Swing height default 0.04 m → foot must lift at least 80% of
        // it to count as a visible march.
        assert!(
            z_span > 0.032,
            "march-in-place: foot z span should be ≈ swing_height (0.04 m), \
             got {z_span} m. Phase generator may be stuck or swing curve \
             collapsed.",
        );
        // Horizontal excursion should be a tiny fraction of swing height.
        assert!(
            max_x_excursion < 1e-3,
            "march-in-place: foot x should stay near nominal (got excursion \
             {max_x_excursion} m). With cmd.vx=1e-6 and T_stance=0.2 s the \
             Raibert step is 1e-7 m — anything larger means a different \
             code path is creeping into the footstep.",
        );
    }
}
