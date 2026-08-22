//! Wire format for **live gait visualization** — one self-contained frame of a
//! generated gait, streamed from a runner (e.g. `go2-gait-runner`) to a viewer
//! (e.g. the `articara` GUI) so the gait can be watched in real time.
//!
//! The transport is left to the caller (the design uses Zenoh); this module
//! only defines the payload and its (de)serialization. Each frame is a full
//! pose, so the channel can be lossy / latest-wins.
//!
//! # Encoding
//! The struct derives `serde::{Serialize, Deserialize}`, so the caller picks
//! the wire encoding: start with JSON / CBOR for readability, switch to a
//! compact binary (`bincode`, …) later for throughput. The transport should
//! tag the payload with its encoding (e.g. Zenoh's `Encoding`) and the version
//! field below guards against schema drift.
//!
//! # Joint order
//! [`GaitVizFrame::joints`] is `slot × (hip, thigh, calf)` with the slot order
//! **FL, FR, RL, RR** (= [`crate::controller`]'s `slot_of`). The viewer maps
//! the 12 angles onto its own model via its detected per-leg joint names.
//!
//! # Publishing
//! `doc/viz_publisher.md` in this repo writes this out for someone implementing
//! a publisher from scratch; the contract itself is:
//!
//! A publisher that only generates a gait sends one stream, on
//! [`VIZ_KEY_PLANNED`]. One that also reads state back sends a second stream on
//! [`VIZ_KEY_MEASURED`], and the pairing is what makes the two useful:
//!
//! - **Same tick, same [`GaitVizFrame::seq`]** for the two frames. The viewer
//!   samples each stream independently, so this is what bounds their skew to
//!   one publish period.
//! - **Distinct keys.** The channel is latest-wins; one key carrying both poses
//!   would have them overwrite each other.
//! - **Model convention, not IK convention.** [`GaitVizFrame::from_output`]
//!   fills the joints in the gait's IK convention; sign-correct them before
//!   publishing (see that method's note). Angles read back from a robot are
//!   already in the model convention and need no correction, only re-ordering
//!   into the slot order above.
//! - **The measured pose is the measured one.** Sending the commanded pose on
//!   the measured stream makes the two bodies coincide, which flatters the
//!   picture and hides where the robot actually went. A viewer that wants them
//!   superimposed re-anchors on its side.
//! - **Publish off the control loop.** Serializing and putting a frame is not
//!   free and a stalled peer blocks; hand frames to a publisher thread over a
//!   bounded channel and drop when it is full.
//! - **Don't publish a pose you haven't measured yet.** Before the first state
//!   read-back there is nothing to send; a zeroed frame renders as a collapsed
//!   robot.

use crate::controller::ControllerOutput;

/// Wire format version. Bump on any incompatible change to [`GaitVizFrame`].
///
/// Purely *additive* fields carrying a `serde(default)` don't count: both
/// directions of a mixed deployment keep decoding (see [`GaitVizFrame::pose_rp`]).
pub const VIZ_FORMAT_VERSION: u32 = 1;

/// Default Zenoh key expression for the **planned** (controller-output) gait
/// stream — what the controller asked the robot to do.
pub const VIZ_KEY_PLANNED: &str = "go2/gait/planned";

/// Default Zenoh key expression for the **measured** stream — what the robot
/// reported back. A publisher with state read-back (a hardware runner) sends
/// both, on the same tick and under the same [`GaitVizFrame::seq`], so a viewer
/// can superimpose command and response.
///
/// The two must stay distinct keys: the channel is latest-wins, so one key
/// carrying both poses would just have them overwrite each other.
pub const VIZ_KEY_MEASURED: &str = "go2/gait/measured";

/// One frame of a generated gait for live visualization.
///
/// Self-contained: carries the full body pose + all joint angles, so a viewer
/// can render it standalone and a lossy transport (latest-wins) is fine.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "viz", derive(serde::Serialize, serde::Deserialize))]
pub struct GaitVizFrame {
    /// Format version (= [`VIZ_FORMAT_VERSION`]); viewers reject mismatches.
    pub version: u32,
    /// Monotonic sequence number — lets the viewer detect loss / reordering.
    pub seq: u64,
    /// Gait time, seconds (since the run started).
    pub t_s: f64,
    /// Body world pose `[x, y, z, yaw]` — metres and radians. `z` is the trunk
    /// height above the ground (the gait integrates only `x, y`).
    pub pose: [f64; 4],
    /// Trunk **roll and pitch** (rad) — the two attitude angles [`Self::pose`]
    /// has no room for.
    ///
    /// `[0, 0]` (level) is the meaningful default, and what a missing field
    /// decodes to: the gait plans a level trunk ([`crate::BodyState`] carries
    /// only position and yaw), so a *planned* stream has no attitude to report
    /// in the first place. A *measured* stream fills these from the IMU, which
    /// is the whole point — trunk tilt is tracking error a viewer wants to see.
    ///
    /// Added after [`VIZ_FORMAT_VERSION`] 1 without bumping it: the field is
    /// `serde(default)`, so a new reader accepts old frames, and an old reader
    /// ignores the new field. Neither side breaks on a mixed deployment.
    #[cfg_attr(feature = "viz", serde(default))]
    pub pose_rp: [f64; 2],
    /// 12 joint angles (rad), slot order **FL, FR, RL, RR** × (hip, thigh,
    /// calf). Intended to be settable directly as a viewer's URDF/model joint
    /// positions. NOTE: [`Self::from_output`] fills these from the controller
    /// in the **gait/IK convention**; a publisher driving a robot model should
    /// sign-correct them to the model convention first (multiply by the
    /// `joint_signs` IK→model table — the same correction the hardware path
    /// applies), otherwise sign-flipped joints (e.g. the knee) render mirrored.
    pub joints: [f64; 12],
    /// Per-slot stance flag (FL, FR, RL, RR); `true` = foot planted. For the
    /// viewer to colour stance vs swing legs.
    pub stance: [bool; 4],
}

impl GaitVizFrame {
    /// Build a frame from a controller tick. `seq` is a monotonic counter,
    /// `t_s` the gait time, and `trunk_z` the body height above the ground
    /// (the controller output carries only the horizontal `x, y`).
    pub fn from_output(seq: u64, t_s: f64, trunk_z: f64, out: &ControllerOutput) -> Self {
        let mut joints = [0.0f64; 12];
        let mut stance = [false; 4];
        for slot in 0..4 {
            let l = &out.legs[slot];
            joints[3 * slot] = l.q_hip;
            joints[3 * slot + 1] = l.q_thigh;
            joints[3 * slot + 2] = l.q_calf;
            stance[slot] = l.phase.is_stance;
        }
        let b = &out.body_state;
        Self {
            version: VIZ_FORMAT_VERSION,
            seq,
            t_s,
            pose: [b.world_position.x, b.world_position.y, trunk_z, b.world_yaw],
            // The gait plans a level trunk; a publisher with a measured
            // attitude overwrites this.
            pose_rp: [0.0, 0.0],
            joints,
            stance,
        }
    }

    /// Whether this frame's [`Self::version`] matches what this build expects.
    pub fn is_compatible(&self) -> bool {
        self.version == VIZ_FORMAT_VERSION
    }
}

#[cfg(all(test, feature = "viz"))]
mod tests {
    use super::*;

    fn frame() -> GaitVizFrame {
        GaitVizFrame {
            version: VIZ_FORMAT_VERSION,
            seq: 7,
            t_s: 0.5,
            pose: [1.0, 2.0, 0.3, 0.4],
            pose_rp: [0.05, -0.02],
            joints: [0.1; 12],
            stance: [true, false, true, false],
        }
    }

    #[test]
    fn attitude_survives_a_round_trip() {
        let json = serde_json::to_vec(&frame()).unwrap();
        let back: GaitVizFrame = serde_json::from_slice(&json).unwrap();
        assert_eq!(back, frame());
    }

    /// A publisher predating `pose_rp` omits the field; it must decode as a
    /// level trunk rather than failing the frame — the mixed-deployment case
    /// the version comment promises.
    #[test]
    fn a_frame_without_attitude_decodes_as_level() {
        let mut value = serde_json::to_value(frame()).unwrap();
        value.as_object_mut().unwrap().remove("pose_rp").unwrap();
        let back: GaitVizFrame = serde_json::from_value(value).unwrap();
        assert_eq!(back.pose_rp, [0.0, 0.0]);
        assert!(back.is_compatible(), "still a version-1 frame");
        assert_eq!(back.pose, frame().pose, "the rest is untouched");
    }

    /// The other direction: a field this build doesn't know about is ignored,
    /// so a newer publisher doesn't break an older viewer.
    #[test]
    fn an_unknown_field_is_ignored() {
        let mut value = serde_json::to_value(frame()).unwrap();
        value.as_object_mut().unwrap().insert(
            "some_future_field".into(),
            serde_json::json!([1, 2, 3]),
        );
        let back: GaitVizFrame = serde_json::from_value(value).unwrap();
        assert_eq!(back, frame());
    }
}
