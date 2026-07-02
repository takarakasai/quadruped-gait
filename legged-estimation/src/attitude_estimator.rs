//! Madgwick attitude estimator (6-axis IMU variant).
//!
//! Implements Sebastian Madgwick's gradient-descent attitude filter for
//! a 6-axis IMU (3-axis accelerometer + 3-axis gyroscope, **no
//! magnetometer**). Reference: *S. Madgwick, "An efficient orientation
//! filter for inertial and inertial/magnetic sensor arrays" (2010)*,
//! IMU-only variant in §3.5.
//!
//! Without a magnetometer the filter has **no absolute heading
//! reference**: roll and pitch are observable from gravity in the
//! accelerometer, but yaw integrates only from the gyroscope and drifts
//! over time. Acceptable for namiashi (and any in-house quadruped that
//! doesn't carry a magnetometer / GPS), but document this limitation
//! clearly in the UI.
//!
//! # Conventions
//!
//! - All vectors are expressed in the IMU body frame (matches what
//!   the host's IMU reading reports).
//! - The estimator's output quaternion `q` is the rotation **from
//!   sensor frame to world (gravity-aligned) frame**, in `(w, x, y, z)`
//!   order.
//! - Gravity convention: world Z up. A stationary IMU on a horizontal
//!   surface reads `accel = [0, 0, +g]` in its body frame; the filter
//!   converges to `q = identity` in that case.
//!
//! # Tuning
//!
//! `beta` controls the gradient-descent step (how aggressively the
//! accel-derived correction is fed back). Madgwick's paper recommends
//! `beta = sqrt(3/4) * gyroMeasError`. Default `0.1` rad/s is a sane
//! starting point for low-noise sims; raise it for noisy real-world
//! gyros at the cost of accelerometer-induced jitter.

use nalgebra::{UnitQuaternion, Quaternion};

/// Madgwick attitude estimator state.
///
/// Construct with [`Self::new`] (identity orientation), then call
/// [`Self::update_imu`] every IMU sample. The current attitude estimate
/// is available via [`Self::quaternion`] / [`Self::euler_zyx`].
#[derive(Clone, Debug)]
pub struct MadgwickAhrs {
    /// Filter gain. Higher = trust accel more, lower = trust gyro more.
    /// Madgwick recommends `sqrt(3/4) * gyro_measurement_error_rad_s`.
    pub beta: f64,
    /// Current estimate: rotation from sensor frame to world.
    q: UnitQuaternion<f64>,
}

impl Default for MadgwickAhrs {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl MadgwickAhrs {
    /// Construct an estimator at identity orientation with the given
    /// `beta` gain (rad/s, see struct-level docs for tuning guidance).
    pub fn new(beta: f64) -> Self {
        Self { beta, q: UnitQuaternion::identity() }
    }

    /// Reset to identity orientation. Call after a known-stationary
    /// pose (e.g. user clicked "calibrate" or after stop / restart of
    /// the sim).
    pub fn reset(&mut self) {
        self.q = UnitQuaternion::identity();
    }

    /// Current attitude estimate: sensor → world rotation.
    pub fn quaternion(&self) -> UnitQuaternion<f64> {
        self.q
    }

    /// Current attitude as ZYX intrinsic Euler angles (roll-around-X,
    /// pitch-around-Y, yaw-around-Z), radians. The `(roll, pitch, yaw)`
    /// triple shown in the UI.
    pub fn euler_zyx(&self) -> (f64, f64, f64) {
        self.q.euler_angles()
    }

    /// Integrate one IMU sample.
    ///
    /// - `gyro`: angular velocity (rad/s) in sensor frame.
    /// - `accel`: proper acceleration (m/s²) in sensor frame —
    ///   includes the gravity reaction, so a stationary IMU should
    ///   report `[0, 0, +g]` along its local +Z axis.
    /// - `dt`: sample period (s); must be positive.
    ///
    /// Skips the accelerometer correction when `|accel|` is near zero
    /// (unsafe to normalise) or matches free-fall — in those frames
    /// the filter falls back to pure gyro integration so the estimate
    /// doesn't snap to a meaningless reference.
    pub fn update_imu(&mut self, gyro: [f64; 3], accel: [f64; 3], dt: f64) {
        if dt <= 0.0 {
            return;
        }

        let (mut q0, mut q1, mut q2, mut q3) = (
            self.q.w,
            self.q.i,
            self.q.j,
            self.q.k,
        );

        let (gx, gy, gz) = (gyro[0], gyro[1], gyro[2]);

        // Quaternion derivative from gyroscope (Hamilton product q * ω/2).
        let mut q_dot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
        let mut q_dot1 = 0.5 * ( q0 * gx + q2 * gz - q3 * gy);
        let mut q_dot2 = 0.5 * ( q0 * gy - q1 * gz + q3 * gx);
        let mut q_dot3 = 0.5 * ( q0 * gz + q1 * gy - q2 * gx);

        // Accel-derived correction. Skip if accel reading is unusable
        // (zero magnitude, free-fall, etc.) — relying on gyro only for
        // this frame is preferable to dividing by ~0.
        let accel_norm_sq = accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2];
        if accel_norm_sq > 1e-6 {
            let inv_norm = 1.0 / accel_norm_sq.sqrt();
            let ax = accel[0] * inv_norm;
            let ay = accel[1] * inv_norm;
            let az = accel[2] * inv_norm;

            // Reference gravity direction in world frame: +Z.
            // Estimated gravity in body frame: rotate (0, 0, 1) by qᵀ.
            //   = 2 * (q0*q2 - q3*q1, q3*q0 + q1*q2, ½ - q1² - q2²)
            //   (Madgwick eq. 25 with ax_ref = 0, ay_ref = 0, az_ref = 1)
            //
            // The Jacobian and gradient below come from minimising the
            // cost function ‖f(q, accel)‖² with f the difference between
            // the body-frame measured gravity and the rotated reference.

            // f = R(q)ᵀ · g_world − accel_unit  (Madgwick eq. 25)
            let f1 = 2.0 * (q1 * q3 - q0 * q2) - ax;
            let f2 = 2.0 * (q0 * q1 + q2 * q3) - ay;
            let f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - az;

            // Jacobian Jᵀ · f → ∇C  (gradient of the cost)
            let s0 = -2.0 * q2 * f1 + 2.0 * q1 * f2;
            let s1 =  2.0 * q3 * f1 + 2.0 * q0 * f2 - 4.0 * q1 * f3;
            let s2 = -2.0 * q0 * f1 + 2.0 * q3 * f2 - 4.0 * q2 * f3;
            let s3 =  2.0 * q1 * f1 + 2.0 * q2 * f2;

            // Normalise the gradient (Madgwick uses the unit gradient
            // direction so beta has consistent units of rad/s).
            let s_norm = (s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3).sqrt();
            if s_norm > 1e-12 {
                let inv_s = 1.0 / s_norm;
                q_dot0 -= self.beta * s0 * inv_s;
                q_dot1 -= self.beta * s1 * inv_s;
                q_dot2 -= self.beta * s2 * inv_s;
                q_dot3 -= self.beta * s3 * inv_s;
            }
        }

        // Integrate.
        q0 += q_dot0 * dt;
        q1 += q_dot1 * dt;
        q2 += q_dot2 * dt;
        q3 += q_dot3 * dt;

        let n = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3).sqrt();
        if n > 1e-12 {
            let inv = 1.0 / n;
            self.q = UnitQuaternion::from_quaternion(Quaternion::new(
                q0 * inv,
                q1 * inv,
                q2 * inv,
                q3 * inv,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Stationary IMU with `accel = [0, 0, +g]` and zero gyro should
    /// converge to identity orientation regardless of the initial state.
    #[test]
    fn stationary_horizontal_converges_to_identity() {
        let mut ahrs = MadgwickAhrs::new(0.5);
        // Start tilted to make the test non-trivial.
        ahrs.q = UnitQuaternion::from_euler_angles(0.5, 0.3, 0.0);
        for _ in 0..10_000 {
            ahrs.update_imu([0.0, 0.0, 0.0], [0.0, 0.0, 9.81], 0.001);
        }
        let (r, p, _y) = ahrs.euler_zyx();
        assert!(r.abs() < 0.01, "roll = {r}");
        assert!(p.abs() < 0.01, "pitch = {p}");
    }

    /// IMU rolled 30° around X: accelerometer reads gravity tilted into
    /// the Y axis. The filter should recover the roll angle.
    #[test]
    fn tilted_30deg_recovers_roll_from_accel() {
        let mut ahrs = MadgwickAhrs::new(0.5);
        let roll = 30.0_f64.to_radians();
        // World gravity is +Z (proper accel = +g). For a body rolled
        // +30° around X, the body's +Y axis tilts toward world +Z, so
        // gravity in the body frame has positive Y component:
        //   accel_body = R(x, roll)ᵀ · (0,0,g)
        //              = (0, +sin(roll)*g, +cos(roll)*g)
        let ax = 0.0;
        let ay = roll.sin() * 9.81;
        let az = roll.cos() * 9.81;
        for _ in 0..5_000 {
            ahrs.update_imu([0.0, 0.0, 0.0], [ax, ay, az], 0.001);
        }
        let (r, p, _y) = ahrs.euler_zyx();
        assert!((r - roll).abs() < 0.02, "roll = {r}, expected {roll}");
        assert!(p.abs() < 0.02, "pitch should stay 0, got {p}");
    }

    /// Pure rotation around Z (yaw) with `accel = [0, 0, g]` (still
    /// upright) — the filter should integrate the gyro reading
    /// without rejecting it (no accel correction in yaw).
    #[test]
    fn pure_yaw_integration_matches_input() {
        let mut ahrs = MadgwickAhrs::new(0.0); // disable accel correction so
                                                // we test pure gyro integration
        let omega_yaw = PI / 2.0;               // 90°/s
        let dt = 0.001;
        for _ in 0..1_000 {
            ahrs.update_imu([0.0, 0.0, omega_yaw], [0.0, 0.0, 9.81], dt);
        }
        let (_r, _p, y) = ahrs.euler_zyx();
        // 1000 samples * 0.001 s = 1 s → 90°
        assert!(
            (y - PI / 2.0).abs() < 0.01,
            "yaw = {y}, expected {}",
            PI / 2.0
        );
    }

    /// Reset puts the estimator back to identity.
    #[test]
    fn reset_clears_state() {
        let mut ahrs = MadgwickAhrs::new(0.1);
        for _ in 0..100 {
            ahrs.update_imu([1.0, 0.5, 0.0], [0.0, 0.0, 9.81], 0.01);
        }
        let (r0, p0, y0) = ahrs.euler_zyx();
        assert!(r0.abs() + p0.abs() + y0.abs() > 0.01);

        ahrs.reset();
        let (r, p, y) = ahrs.euler_zyx();
        assert_eq!((r, p, y), (0.0, 0.0, 0.0));
    }

    /// Accel reading of zero (free-fall) should not crash; filter
    /// continues with gyro-only integration.
    #[test]
    fn freefall_accel_does_not_panic() {
        let mut ahrs = MadgwickAhrs::new(0.1);
        for _ in 0..100 {
            ahrs.update_imu([0.0, 0.0, 0.1], [0.0, 0.0, 0.0], 0.001);
        }
        // Reaches some non-NaN attitude.
        let (r, p, y) = ahrs.euler_zyx();
        assert!(r.is_finite() && p.is_finite() && y.is_finite());
    }
}
