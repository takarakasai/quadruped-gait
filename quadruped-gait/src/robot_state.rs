//! Platform boundary for the WBC runtime: the source of measured robot
//! state (base pose + joint encoders).
//!
//! The whole-body-control pipeline needs the robot's measured base pose and
//! per-joint `(q, q̇)` every tick. In simulation those come from MuJoCo
//! ground truth; on hardware they come from the SDK's `LowState` (IMU +
//! motor encoders, with a state estimator supplying the base linear terms).
//! Abstracting the read side behind this trait lets the *same* pipeline run
//! against either platform — the caller just hands it a `&impl
//! RobotStateSource`.
//!
//! Queries are by the model's own link / joint **names** (the same strings
//! the kinematics config and `misarta` model use), so no index bookkeeping
//! crosses the boundary. Every method returns `Option` and yields `None`
//! for a name the platform doesn't know, leaving the pipeline to fall back
//! to its cached / nominal value rather than panicking.

use nalgebra::UnitQuaternion;

/// Read-side platform boundary for the WBC runtime. Implemented by the
/// simulator in tests (`MujocoSim`) and by the hardware SDK adapter on the
/// real robot.
pub trait RobotStateSource {
    /// World-frame position `[x, y, z]` (m) of the named body/link, or
    /// `None` if the platform has no such link.
    fn body_world_position(&self, link: &str) -> Option<[f64; 3]>;

    /// World-frame orientation (unit quaternion) of the named body/link,
    /// or `None` if the platform has no such link.
    fn body_world_orientation(&self, link: &str) -> Option<UnitQuaternion<f64>>;

    /// Measured `(q, q̇)` for the named joint (rad, rad/s), or `None` if the
    /// platform has no such joint.
    fn joint_q_qd(&self, joint: &str) -> Option<(f64, f64)>;
}
