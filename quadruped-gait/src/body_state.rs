//! Body pose integrator (open-loop).
//!
//! Tracks the world-frame body pose by Euler-integrating the body-frame
//! velocity command. Yaw is integrated directly from `wz`; the linear
//! displacement projects through the current yaw at the time of integration.
//!
//! The gait controller does not require this for its core operation —
//! footstep planning runs entirely in body frame — but downstream
//! consumers (UI overlay, odometry, scripts that want to know how far
//! the robot walked) benefit from a clean integrator.

use nalgebra::Vector3;

use crate::config::VelocityCmd;

/// Open-loop world-frame body pose. Updated by the controller each tick
/// from the velocity command. Initialised at the world origin facing +x.
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyState {
    /// World-frame body position (m).
    pub world_position: Vector3<f64>,
    /// Body yaw about world Z (rad), counter-clockwise viewed from above.
    pub world_yaw: f64,
}

impl BodyState {
    pub const fn new() -> Self {
        Self {
            world_position: Vector3::new(0.0, 0.0, 0.0),
            world_yaw: 0.0,
        }
    }

    /// Integrate the body-frame command for `dt` seconds. Linear motion is
    /// rotated into world frame by the *current* yaw (Euler step — fine
    /// for the small dt used by the gait controller).
    pub fn integrate(&mut self, cmd: &VelocityCmd, dt: f64) {
        let cos_y = self.world_yaw.cos();
        let sin_y = self.world_yaw.sin();
        let dx_world = (cmd.vx * cos_y - cmd.vy * sin_y) * dt;
        let dy_world = (cmd.vx * sin_y + cmd.vy * cos_y) * dt;
        self.world_position.x += dx_world;
        self.world_position.y += dy_world;
        self.world_yaw += cmd.wz * dt;
    }

    /// Reset to identity (origin, zero yaw).
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn forward_walk_matches_distance() {
        let mut bs = BodyState::new();
        let cmd = VelocityCmd { vx: 0.5, vy: 0.0, wz: 0.0 };
        for _ in 0..1000 {
            bs.integrate(&cmd, 0.001);
        }
        // 0.5 m/s · 1.0 s = 0.5 m
        assert_relative_eq!(bs.world_position.x, 0.5, epsilon = 1e-9);
        assert_relative_eq!(bs.world_position.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(bs.world_yaw, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn pure_yaw_rotates_in_place() {
        let mut bs = BodyState::new();
        let cmd = VelocityCmd { vx: 0.0, vy: 0.0, wz: 1.0 };
        for _ in 0..1000 {
            bs.integrate(&cmd, 0.001);
        }
        assert_relative_eq!(bs.world_yaw, 1.0, epsilon = 1e-9);
        assert_relative_eq!(bs.world_position.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(bs.world_position.y, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn forward_then_left_curve() {
        // Step 1: walk forward 1s @ 0.5 m/s → x = 0.5
        // Step 2: yaw +π/2 instantly, walk forward 1s @ 0.5 m/s → y += 0.5
        let mut bs = BodyState::new();
        let fwd = VelocityCmd { vx: 0.5, vy: 0.0, wz: 0.0 };
        for _ in 0..1000 {
            bs.integrate(&fwd, 0.001);
        }
        bs.world_yaw = std::f64::consts::FRAC_PI_2;
        for _ in 0..1000 {
            bs.integrate(&fwd, 0.001);
        }
        assert_relative_eq!(bs.world_position.x, 0.5, epsilon = 1e-6);
        assert_relative_eq!(bs.world_position.y, 0.5, epsilon = 1e-6);
    }
}
