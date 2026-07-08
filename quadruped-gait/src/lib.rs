//! Quadruped gait generation library.
//!
//! Designed as a Rust port of the open-source [CHAMP](https://github.com/chvmp/champ)
//! controller's core ideas, adapted to articara's existing kinematics
//! library ([`misarta`]) so the same robot description used for IK and
//! dynamics drives the gait IK as well.
//!
//! # Pipeline
//!
//! ```text
//! velocity command
//!        │
//!        ▼
//! ┌──────────────┐
//! │ BodyState    │  integrate vx/vy/wz → pose target
//! └──────────────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ PhaseGen     │  per-leg stance/swing fraction (Trot)
//! └──────────────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ Footstep     │  Raibert heuristic → world-frame foot target
//! └──────────────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ Trajectory   │  Bezier swing curve / linear stance line
//! └──────────────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ Leg IK       │  3-DOF (Roll-Pitch-Pitch) analytical IK
//! └──────────────┘
//!        │
//!        ▼
//! 12 joint targets → MuJoCo controller
//! ```
//!
//! # Scope (v0.1)
//!
//! - **Trot only**. Walk / Pace / Bound / Gallop are scaffolded but not yet
//!   implemented — the [`GaitType`] enum carries them as a future-friendly
//!   hook.
//! - **Hip-Thigh-Calf RPP morphology only** (the standard quadruped layout
//!   shared by Mini Pupper, A1, Spot Mini, Aliengo, Solo, …).
//! - **Open-loop body trajectory** integrated from the velocity command;
//!   no IMU/encoder feedback yet.
//!
//! See `tests/` for unit-tested behaviour and the `articara` GUI for an
//! interactive driver.

// ── Public modules (the front door, together with the re-exports below) ──

/// Metadata-driven experimental research knobs (see [`exp::ExpKey`]).
pub mod exp;
/// Live gait-visualization wire format (see [`viz::GaitVizFrame`]).
pub mod viz;
#[cfg(feature = "viz-sub")]
pub mod viz_sub;
/// Hierarchical whole-body-control QP solver.
pub mod wbc;

// ── Internal modules ──
//
// Private on purpose: consumers (the articara GUI, go2-gait-runner,
// legged-estimation) go through the curated re-exports below, so the
// solver / controller internals can be reshaped without breaking them.

mod async_solver;
mod autodetect;
mod body_state;
mod centroidal_controller;
mod centroidal_mpc;
mod config;
mod controller;
mod footstep;
mod full_centroidal_controller;
mod full_centroidal_mpc;
mod generator;
mod ik;
mod linear_crawl;
mod mpc_controller;
mod mpc_reference;
mod phase;
mod srbd_mpc;
mod swing_traj;

// ── Curated re-exports ──

pub use autodetect::{auto_detect_kinematics_config, auto_detect_leg_kinematics, joint_signs};
pub use config::{
    GaitConfig, GaitType, KinematicsConfig, KneePattern, LegId, LegKinematics,
    VelocityCmd, DEFAULT_FOOT_LINKS,
};
// Existing CHAMP-derived controller. Re-exported under both the legacy
// name (`GaitController`) and the new explicit name (`ChampGaitController`)
// so older callers keep working while new code can name the choice.
pub use controller::{ControllerOutput, GaitController, GaitController as ChampGaitController, LegOutput};
pub use exp::{
    format_presets, load_presets, parse_presets, save_presets, upsert_preset, ExpError, ExpKey,
    ExpKind, ExpPreset, ExpValue,
};
pub use generator::{AnyGaitController, GaitGenerator, GaitMode};
pub use ik::{foot_jacobian_body, forward_leg_kinematics, solve_leg_ik, LegIkSolution};
pub use mpc_controller::capture_point_step;
pub use mpc_reference::JointReference;
pub use phase::ContactDrivenPhase;
// MPC tuning configs + the solution / prediction types the GUI reads
// back for overlays. The solvers themselves (SrbdMpc / CentroidalMpc /
// FullCentroidalMpc) and their input / reference / schedule types are
// internal.
pub use centroidal_mpc::{predicted_base_accel_world_centroidal, CentroidalMpcConfig};
pub use full_centroidal_controller::GoalPoseWorld;
pub use full_centroidal_mpc::FullCentroidalMpcConfig;
pub use srbd_mpc::{predicted_base_accel_world, MpcSolution, SrbdMpcConfig, SrbdState};
