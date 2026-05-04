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

pub mod body_state;
pub mod config;
pub mod controller;
pub mod footstep;
pub mod generator;
pub mod ik;
pub mod mpc_controller;
pub mod phase;
pub mod swing_traj;

pub use body_state::BodyState;
pub use config::{
    GaitConfig, GaitType, KinematicsConfig, KneePattern, LegId, LegKinematics,
    VelocityCmd, DEFAULT_FOOT_LINKS,
};
// Existing CHAMP-derived controller. Re-exported under both the legacy
// name (`GaitController`) and the new explicit name (`ChampGaitController`)
// so older callers keep working while new code can name the choice.
pub use controller::{ControllerOutput, GaitController, GaitController as ChampGaitController, LegOutput};
pub use footstep::{compute_footstep, Footstep};
pub use generator::{AnyGaitController, GaitGenerator, GaitMode};
pub use ik::{solve_leg_ik, LegIkSolution};
pub use mpc_controller::MpcGaitController;
pub use phase::{PhaseGenerator, PhaseState};
pub use swing_traj::{stance_position, swing_position};
