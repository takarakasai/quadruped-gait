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

fn slot_of(id: LegId) -> usize {
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
                    self.cfg.swing_height_m,
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
    fn forward_walk_advances_body_position() {
        let mut ctrl = GaitController::new(GaitConfig::trot(), build_kin());
        ctrl.set_velocity_cmd(VelocityCmd { vx: 0.3, ..Default::default() });
        for _ in 0..500 {
            ctrl.tick(0.002); // 1 second total
        }
        // 0.3 m/s for 1s → ~0.3 m
        assert_relative_eq!(ctrl.body_state().world_position.x, 0.3, epsilon = 1e-9);
    }
}
