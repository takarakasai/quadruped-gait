//! Open-loop Raibert-style footstep planner.
//!
//! Given a velocity command and a leg's nominal foot position, computes the
//! pair `(lift_off, touch_down)` that the leg's foot should travel between
//! during the cycle. The line is centred on the nominal foot pose so the
//! foot appears stationary in body frame at the cycle midpoint.
//!
//! # Geometry
//!
//! With body linear velocity `v_body = (vx, vy, 0)` and yaw rate
//! `ω = (0, 0, wz)`, the velocity at the leg's hip in body frame is
//!
//! ```text
//! v_hip = v_body + ω × p_hip
//! ```
//!
//! where `p_hip` is the hip's offset from body origin. The foot must stay
//! planted in the world while the body moves, so in the body frame the
//! foot moves at `-v_hip` during stance. Spanning half the stance
//! duration on each side of the nominal gives:
//!
//! ```text
//! touch_down = nominal_foot + 0.5 · T_stance · v_hip
//! lift_off   = nominal_foot − 0.5 · T_stance · v_hip
//! ```
//!
//! The result is clipped to `gait.max_step_length_m / 2` per side so the
//! controller can't ask the IK for a foot beyond the leg's reach.
//!
//! # Closed-loop extension (future)
//!
//! Real CHAMP uses a feedback term `√(h/g) · (v_actual − v_cmd)` from a
//! state estimator. Phase 2 is open-loop — the host can plug in a
//! corrected `cmd` later by closing the loop externally if desired.

use nalgebra::Vector3;

use crate::config::{GaitConfig, LegKinematics, VelocityCmd};

/// Result of one footstep computation: stance-line endpoints in body frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footstep {
    pub lift_off: Vector3<f64>,
    pub touch_down: Vector3<f64>,
}

impl Footstep {
    /// Position along the stance line at sub-phase `frac ∈ [0, 1]`.
    /// `frac = 0` is **touch-down** (the foot has just landed at the front
    /// of the stride), `frac = 1` is **lift-off** (the foot is about to
    /// leave the ground at the back of the stride). This direction —
    /// front-to-back in body frame — is the one that pushes the body
    /// forward when contact is present, and matches the swing's
    /// lift_off → touch_down direction so there's no discontinuity at
    /// the stance/swing boundary.
    pub fn stance_at(&self, frac: f64) -> Vector3<f64> {
        let f = frac.clamp(0.0, 1.0);
        self.touch_down * (1.0 - f) + self.lift_off * f
    }
}

/// Compute the open-loop Raibert footstep for one leg.
///
/// The function never fails — extreme commands are clamped via
/// `gait.max_step_length_m`. Pass `vx = 0` for "stand still" semantics
/// (the planner returns `lift_off = touch_down = nominal_foot`).
pub fn compute_footstep(
    kin: &LegKinematics,
    gait: &GaitConfig,
    cmd: &VelocityCmd,
) -> Footstep {
    let stance_duration = gait.cycle_period_s * gait.duty_factor;
    // Hip linear velocity in body frame, including the yaw-induced part.
    let v_body = Vector3::new(cmd.vx, cmd.vy, 0.0);
    let omega = Vector3::new(0.0, 0.0, cmd.wz);
    let v_hip = v_body + omega.cross(&kin.hip_offset);
    let mut half = v_hip * (0.5 * stance_duration);
    // Clamp at the configured maximum step length (sphere of radius
    // max_step / 2 around the nominal foot).
    let max_half = 0.5 * gait.max_step_length_m;
    let mag = half.norm();
    if mag > max_half && mag > 0.0 {
        half *= max_half / mag;
    }
    Footstep {
        lift_off: kin.nominal_foot_body - half,
        touch_down: kin.nominal_foot_body + half,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LegId;
    use approx::assert_relative_eq;

    fn fl_kin() -> LegKinematics {
        LegKinematics::new(
            LegId::FL,
            "FL_hip".into(),
            "FL_thigh".into(),
            "FL_calf".into(),
            "FL_foot".into(),
            Vector3::new(0.18, 0.05, 0.0),
            0.04,
            0.18,
            0.18,
        )
    }

    fn fr_kin() -> LegKinematics {
        LegKinematics::new(
            LegId::FR,
            "FR_hip".into(),
            "FR_thigh".into(),
            "FR_calf".into(),
            "FR_foot".into(),
            Vector3::new(0.18, -0.05, 0.0),
            0.04,
            0.18,
            0.18,
        )
    }

    #[test]
    fn forward_only_centers_step_around_nominal() {
        let kin = fl_kin();
        let gait = GaitConfig::trot();
        let cmd = VelocityCmd { vx: 0.3, vy: 0.0, wz: 0.0 };
        let fs = compute_footstep(&kin, &gait, &cmd);
        // step span = vx · T_stance = 0.3 · 0.2 = 0.06 m → ±0.03 m around nominal
        let expected_half = 0.03;
        let half_dx = fs.touch_down.x - kin.nominal_foot_body.x;
        assert_relative_eq!(half_dx, expected_half, epsilon = 1e-9);
        let half_dx_lo = kin.nominal_foot_body.x - fs.lift_off.x;
        assert_relative_eq!(half_dx_lo, expected_half, epsilon = 1e-9);
        // y / z untouched
        assert_relative_eq!(fs.touch_down.y, kin.nominal_foot_body.y);
        assert_relative_eq!(fs.touch_down.z, kin.nominal_foot_body.z);
    }

    #[test]
    fn zero_cmd_collapses_to_nominal() {
        let kin = fl_kin();
        let gait = GaitConfig::trot();
        let fs = compute_footstep(&kin, &gait, &VelocityCmd::zero());
        for ax in 0..3 {
            assert_relative_eq!(fs.lift_off[ax], kin.nominal_foot_body[ax]);
            assert_relative_eq!(fs.touch_down[ax], kin.nominal_foot_body[ax]);
        }
    }

    #[test]
    fn yaw_rate_creates_opposite_lateral_steps() {
        // Pure yaw command: turning left (wz > 0) means front-left hip
        // moves forward + outward, front-right hip moves forward + inward.
        // The y-component of v_hip = wz · hip_offset.x · (-direction).
        // Actually ω × p_hip = (0,0,wz) × (px, py, 0) = (-wz·py, wz·px, 0).
        // So for FL (px=0.18, py=0.05):  v_hip_y = wz · 0.18 = 0.18·wz (positive when turning left).
        // For FR (px=0.18, py=-0.05):    v_hip_y = wz · 0.18 = same sign! Both fronts move +y for left turn.
        // What differs is v_hip_x: -wz·py → FL gets -wz·0.05 (back), FR gets +wz·0.05 (forward).
        let gait = GaitConfig::trot();
        let cmd = VelocityCmd { vx: 0.0, vy: 0.0, wz: 1.0 };
        let fs_fl = compute_footstep(&fl_kin(), &gait, &cmd);
        let fs_fr = compute_footstep(&fr_kin(), &gait, &cmd);
        let dx_fl = fs_fl.touch_down.x - fl_kin().nominal_foot_body.x;
        let dx_fr = fs_fr.touch_down.x - fr_kin().nominal_foot_body.x;
        // FL goes backward, FR goes forward when turning left in place.
        assert!(dx_fl < -1e-6, "FL should step backward when turning left, got {dx_fl}");
        assert!(dx_fr > 1e-6, "FR should step forward when turning left, got {dx_fr}");
    }

    #[test]
    fn extreme_cmd_clamped_to_max_step() {
        let kin = fl_kin();
        // Default max_step_length_m = 0.10
        let gait = GaitConfig::trot();
        let cmd = VelocityCmd { vx: 5.0, vy: 0.0, wz: 0.0 }; // 5 m/s → step would be 1 m
        let fs = compute_footstep(&kin, &gait, &cmd);
        let half_x = fs.touch_down.x - kin.nominal_foot_body.x;
        assert_relative_eq!(half_x, gait.max_step_length_m * 0.5, epsilon = 1e-9);
    }

    #[test]
    fn stance_at_endpoints_exact() {
        // Stance starts at touch_down (foot just landed) and ends at
        // lift_off (foot about to leave). This direction is what pushes
        // the body forward — the controller had this backwards in the
        // first Phase 2 cut.
        let kin = fl_kin();
        let gait = GaitConfig::trot();
        let cmd = VelocityCmd { vx: 0.3, vy: 0.0, wz: 0.0 };
        let fs = compute_footstep(&kin, &gait, &cmd);
        let p0 = fs.stance_at(0.0);
        let p1 = fs.stance_at(1.0);
        for ax in 0..3 {
            assert_relative_eq!(p0[ax], fs.touch_down[ax], epsilon = 1e-9);
            assert_relative_eq!(p1[ax], fs.lift_off[ax], epsilon = 1e-9);
        }
    }

    #[test]
    fn stance_swing_transition_is_continuous() {
        // The end of stance must coincide with the start of swing — same
        // foot position in body frame — otherwise the controller emits
        // a step-change in the position target every cycle, generating
        // a huge transient torque (and visually-broken trot motion).
        let kin = fl_kin();
        let gait = GaitConfig::trot();
        let cmd = VelocityCmd { vx: 0.3, vy: 0.0, wz: 0.0 };
        let fs = compute_footstep(&kin, &gait, &cmd);
        let stance_end = fs.stance_at(1.0);
        let swing_start = crate::swing_traj::swing_position(
            fs.lift_off,
            fs.touch_down,
            gait.swing_height_m,
            0.0,
        );
        for ax in 0..3 {
            assert_relative_eq!(stance_end[ax], swing_start[ax], epsilon = 1e-9);
        }

        // Symmetric check: swing's end must coincide with stance's start.
        let swing_end = crate::swing_traj::swing_position(
            fs.lift_off,
            fs.touch_down,
            gait.swing_height_m,
            1.0,
        );
        let stance_start = fs.stance_at(0.0);
        for ax in 0..3 {
            assert_relative_eq!(swing_end[ax], stance_start[ax], epsilon = 1e-9);
        }
    }
}
