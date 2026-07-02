//! Kinematics-based body velocity / position estimator for quadrupeds.
//!
//! Real legged platforms can't use the simulator's `body cvel` /
//! `xpos` oracle — they have to *infer* their body pose from leg
//! encoders + IMU. The standard recipe (used by MIT-Cheetah,
//! ANYmal, Pronto) is:
//!
//! ```text
//! foot pinned to ground (stance) ⇒ d/dt p_foot_world = 0
//! p_foot_world = p_body_world + R · p_foot_body
//! 0 = v_body_world + ω_world × R · p_foot_body
//!     + R · (J_leg · q̇_leg)
//! ⇒ v_body_world = -R · (J_leg · q̇_leg)  -  ω_world × R · p_foot_body
//! ```
//!
//! Equivalently in body frame:
//!
//! ```text
//! v_body_body = -J_leg · q̇_leg  -  ω_body × p_foot_body
//! ```
//!
//! We average the per-stance-leg estimates (uniform weight; future
//! work could weight by GRF magnitude / contact confidence). Position
//! is `∫ v dt` — drifts slowly, bounded only by encoder + gyro bias
//! and stance-foot-slip.
//!
//! The estimator only handles **horizontal velocity / position**
//! (yaw-only world model). Vertical motion is inferred but not used
//! by the SRBD MPC, which regulates body z to a constant nominal
//! height.

use nalgebra as na;

use quadruped_gait::{foot_jacobian_body, forward_leg_kinematics, KinematicsConfig};

/// Stateful leg-odometry estimator.
#[derive(Clone, Debug)]
pub struct LegOdometry {
    /// Integrated body world-frame position (m). Set externally via
    /// [`Self::set_position`] when the host wants to seed the estimator
    /// (e.g. at MuJoCo sim start).
    position_world: na::Vector3<f64>,
    /// Last estimated body world-frame velocity (m/s). Exposed so the
    /// UI can plot it / compare with MuJoCo's `cvel`.
    last_velocity_world: na::Vector3<f64>,
}

impl LegOdometry {
    pub fn new() -> Self {
        Self {
            position_world: na::Vector3::zeros(),
            last_velocity_world: na::Vector3::zeros(),
        }
    }

    /// Reset both the position and the cached velocity to zero.
    pub fn reset(&mut self) {
        self.position_world = na::Vector3::zeros();
        self.last_velocity_world = na::Vector3::zeros();
    }

    /// Seed the estimator's position with a known starting value.
    /// Useful when the sim starts the body at non-origin or after a
    /// ground-truth re-localisation event.
    pub fn set_position(&mut self, p: na::Vector3<f64>) {
        self.position_world = p;
    }

    pub fn position_world(&self) -> na::Vector3<f64> {
        self.position_world
    }

    pub fn last_velocity_world(&self) -> na::Vector3<f64> {
        self.last_velocity_world
    }

    /// One sim-time step. Inputs:
    /// - `kin`: leg kinematic constants (built once at sim start).
    /// - `legs`: per-slot tuple in canonical FL/FR/RL/RR order with
    ///   `(q_hip, q_thigh, q_calf, q̇_hip, q̇_thigh, q̇_calf, is_stance)`.
    ///   `q*` and `q̇*` are in **IK convention** — caller converts from
    ///   URDF axes via `GaitController::joint_signs` before calling.
    /// - `omega_body`: body angular velocity (rad/s). For yaw-only
    ///   models pass `(0, 0, ω_z)` from the gyro / Madgwick.
    /// - `yaw_world`: current world yaw used to rotate the body-frame
    ///   estimate into world frame for integration.
    /// - `dt`: time elapsed since the previous call (s).
    ///
    /// When no leg is in stance, the previous velocity is held and
    /// position drifts linearly (free flight assumption). This matches
    /// what real-robot estimators do during the brief flight phase of
    /// a trot — encoder data alone can't observe body motion when
    /// nothing's on the ground.
    pub fn update(
        &mut self,
        kin: &KinematicsConfig,
        legs: [(f64, f64, f64, f64, f64, f64, bool); 4],
        omega_body: na::Vector3<f64>,
        yaw_world: f64,
        dt: f64,
    ) {
        if dt <= 0.0 {
            return;
        }
        let leg_kin = kin.legs();
        let mut v_body_sum = na::Vector3::zeros();
        let mut count = 0usize;
        for slot in 0..4 {
            let (q_h, q_t, q_c, qd_h, qd_t, qd_c, stance) = legs[slot];
            if !stance {
                continue;
            }
            // Foot position + Jacobian in body frame at the current
            // joint configuration.
            let p_foot = forward_leg_kinematics(leg_kin[slot], q_h, q_t, q_c);
            let j = foot_jacobian_body(leg_kin[slot], q_h, q_t, q_c);
            let qd = na::Vector3::new(qd_h, qd_t, qd_c);
            let v_foot_kinematic_body = j * qd;
            // Stance constraint solved for body-frame velocity:
            //   v_body_body = -J·q̇_leg − ω_body × p_foot_body
            let v_body_body = -v_foot_kinematic_body - omega_body.cross(&p_foot);
            v_body_sum += v_body_body;
            count += 1;
        }

        if count > 0 {
            let v_body_avg = v_body_sum / count as f64;
            // Rotate body → world (yaw-only).
            let (s, c) = yaw_world.sin_cos();
            self.last_velocity_world = na::Vector3::new(
                c * v_body_avg.x - s * v_body_avg.y,
                s * v_body_avg.x + c * v_body_avg.y,
                v_body_avg.z,
            );
        }
        // No-stance ticks hold last_velocity_world unchanged → free
        // flight integration. See doc comment.
        self.position_world += self.last_velocity_world * dt;
    }
}

impl Default for LegOdometry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_kin() -> KinematicsConfig {
        let leg = |id: quadruped_gait::LegId, hip: na::Vector3<f64>| {
            let mut k = quadruped_gait::LegKinematics::new(
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
            fl: leg(quadruped_gait::LegId::FL, na::Vector3::new(0.18, 0.05, 0.0)),
            fr: leg(quadruped_gait::LegId::FR, na::Vector3::new(0.18, -0.05, 0.0)),
            rl: leg(quadruped_gait::LegId::RL, na::Vector3::new(-0.18, 0.05, 0.0)),
            rr: leg(quadruped_gait::LegId::RR, na::Vector3::new(-0.18, -0.05, 0.0)),
        }
    }

    /// Stationary body, all four feet planted, zero joint velocities →
    /// estimator should report zero body velocity and zero drift.
    #[test]
    fn static_stance_produces_zero_velocity() {
        let kin = build_kin();
        let mut odo = LegOdometry::new();
        // q at IK-convention "neutral pose" (hip 0, thigh 0.5 rad fwd,
        // calf -1.0 rad — arbitrary but fixed). Joint velocities zero.
        let legs = [
            (0.0, 0.5, -1.0, 0.0, 0.0, 0.0, true),
            (0.0, 0.5, -1.0, 0.0, 0.0, 0.0, true),
            (0.0, 0.5, -1.0, 0.0, 0.0, 0.0, true),
            (0.0, 0.5, -1.0, 0.0, 0.0, 0.0, true),
        ];
        for _ in 0..100 {
            odo.update(&kin, legs, na::Vector3::zeros(), 0.0, 0.002);
        }
        approx::assert_relative_eq!(
            odo.last_velocity_world().norm(),
            0.0,
            epsilon = 1e-12,
        );
        approx::assert_relative_eq!(
            odo.position_world().norm(),
            0.0,
            epsilon = 1e-12,
        );
    }

    /// Apply a uniform world-frame body velocity by rotating each leg's
    /// joints such that `J · q̇ = -v_body_body` (foot stays planted while
    /// body moves). Estimator must recover the imposed velocity.
    #[test]
    fn matches_imposed_body_velocity_via_joint_rates() {
        let kin = build_kin();
        let mut odo = LegOdometry::new();
        // Pick a forward-only body velocity in body frame.
        let v_body_target = na::Vector3::new(0.3_f64, 0.0, 0.0);
        // For each leg, solve q̇ such that J · q̇ = -v_body_target. We
        // use the analytical Jacobian at the neutral pose.
        let q_neutral = (0.0_f64, 0.5_f64, -1.0_f64);
        let leg_kin = kin.legs();
        let mut legs = [(q_neutral.0, q_neutral.1, q_neutral.2, 0.0, 0.0, 0.0, true); 4];
        for slot in 0..4 {
            let j = foot_jacobian_body(leg_kin[slot], q_neutral.0, q_neutral.1, q_neutral.2);
            // q̇ = J⁻¹ · (−v_body)
            let qd = j
                .try_inverse()
                .expect("Jacobian invertible at neutral pose")
                * (-v_body_target);
            legs[slot].3 = qd[0];
            legs[slot].4 = qd[1];
            legs[slot].5 = qd[2];
        }
        odo.update(&kin, legs, na::Vector3::zeros(), 0.0, 0.002);
        let v_est = odo.last_velocity_world();
        approx::assert_relative_eq!(v_est.x, 0.3, epsilon = 1e-9);
        approx::assert_relative_eq!(v_est.y, 0.0, epsilon = 1e-9);
        approx::assert_relative_eq!(v_est.z, 0.0, epsilon = 1e-9);
    }

    /// All-swing tick (no stance leg) must hold the previous velocity
    /// estimate and integrate position linearly — this is the "free
    /// flight" model that real estimators apply during the brief
    /// trot flight phase.
    #[test]
    fn no_stance_holds_velocity_and_drifts_position() {
        let kin = build_kin();
        let mut odo = LegOdometry::new();
        // Seed last_velocity by running one stance step at 0.3 m/s.
        let q_neutral = (0.0_f64, 0.5_f64, -1.0_f64);
        let leg_kin = kin.legs();
        let mut legs = [(q_neutral.0, q_neutral.1, q_neutral.2, 0.0, 0.0, 0.0, true); 4];
        for slot in 0..4 {
            let j = foot_jacobian_body(leg_kin[slot], q_neutral.0, q_neutral.1, q_neutral.2);
            let qd = j.try_inverse().unwrap() * na::Vector3::new(-0.3, 0.0, 0.0);
            legs[slot].3 = qd[0];
            legs[slot].4 = qd[1];
            legs[slot].5 = qd[2];
        }
        odo.update(&kin, legs, na::Vector3::zeros(), 0.0, 0.002);
        let p_before = odo.position_world();
        let v_before = odo.last_velocity_world();

        // All-swing tick (no stance, no joint motion): velocity should
        // be unchanged; position should advance by v · dt.
        let swing_legs = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false); 4];
        odo.update(&kin, swing_legs, na::Vector3::zeros(), 0.0, 0.01);
        approx::assert_relative_eq!(
            (odo.last_velocity_world() - v_before).norm(),
            0.0,
            epsilon = 1e-12,
        );
        approx::assert_relative_eq!(
            odo.position_world().x - p_before.x,
            v_before.x * 0.01,
            epsilon = 1e-12,
        );
    }
}
