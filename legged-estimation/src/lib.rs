//! State estimation for legged robots.
//!
//! Pure estimation logic with no simulator or GUI dependency — hosts feed
//! measurements in and consume the estimated state:
//!
//! - [`linear_kalman`] — 18-state linear Kalman filter over
//!   `[body_pos; body_vel; foot_pos_world]`, a port of `legged_control`'s
//!   `KalmanFilterEstimate`.
//! - [`attitude_estimator`] — complementary-filter IMU attitude estimation.
//! - [`leg_odometry`] — contact-based body-velocity odometry from leg
//!   kinematics (via `quadruped-gait`).
//!
//! Extracted from articara (see its `doc/refactor_20260702.md` §4.2, B1)
//! so hardware runners (e.g. go2-gait-runner) can estimate state without
//! pulling in the editor.

pub mod attitude_estimator;
pub mod leg_odometry;
pub mod linear_kalman;

pub use attitude_estimator::*;
pub use leg_odometry::*;
pub use linear_kalman::{LinearKalmanEstimator, LinearKalmanInputs, LinearKalmanOutput};
