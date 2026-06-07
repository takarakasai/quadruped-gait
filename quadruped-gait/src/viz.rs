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

use crate::controller::ControllerOutput;

/// Wire format version. Bump on any incompatible change to [`GaitVizFrame`].
pub const VIZ_FORMAT_VERSION: u32 = 1;

/// Default Zenoh key expression for the **planned** (controller-output) gait
/// stream. A future *measured* stream would use `go2/gait/measured`.
pub const VIZ_KEY_PLANNED: &str = "go2/gait/planned";

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
            joints,
            stance,
        }
    }

    /// Whether this frame's [`Self::version`] matches what this build expects.
    pub fn is_compatible(&self) -> bool {
        self.version == VIZ_FORMAT_VERSION
    }
}
