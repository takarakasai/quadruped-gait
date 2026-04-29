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

pub mod config;
pub mod ik;
pub mod phase;
pub mod swing_traj;

pub use config::{
    GaitConfig, GaitType, KinematicsConfig, LegId, LegKinematics, VelocityCmd,
};
pub use ik::{solve_leg_ik, LegIkSolution};
pub use phase::{PhaseGenerator, PhaseState};
pub use swing_traj::{stance_position, swing_position};
