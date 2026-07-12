//! Hierarchical Whole-Body Control (HoQp) layer.
//!
//! Replaces the simple `τ_ff = -J^T·f_GRF` feedforward with a 3-priority
//! Hierarchical Quadratic Program that solves for joint torques while
//! **strictly enforcing** physical constraints (floating-base equation
//! of motion, friction cone, torque limits, no-contact-motion of stance
//! feet) and **softly tracking** the MPC's reference (base acceleration,
//! swing-leg trajectory, contact-force regularisation).
//!
//! Following [Kim 2014](https://doi.org/10.1109/TRO.2013.2293057) and
//! [Bouyarmane 2018](https://hal.archives-ouvertes.fr/hal-01613147),
//! the structure mirrors `legged_control`'s
//! [`legged_wbc::Wbc`](https://github.com/qiayuanliao/legged_control)
//! line-by-line so the well-tested formulation can be carried over.
//!
//! Decision variable layout per tick:
//!
//! ```text
//! x = [ q̈   |  f_GRF  |  τ ]   ∈  R^(nv + 3·nc + na)
//! ```
//!
//! where `nv` is the number of generalised velocities (6 + actuated
//! joints), `nc` is the number of 3-DoF contacts, and `na` the number
//! of actuated joints.
//!
//! See [`doc/mpc_wbc_gait_control.md`](../../../doc/mpc_wbc_gait_control.md)
//! for the full design doc.

pub mod solver;
pub mod tasks;
mod wbc;

// The HoQP core (dims / HoQp / Task / WarmStart) now lives in the
// standalone, model-agnostic `misa-wbc` crate; this module keeps the
// quadruped-specific task formulations (`tasks`) and the assembled
// per-tick solve (`wbc`). Re-exported so the `tasks` builders and
// downstream callers keep referring to `wbc::{Task, WbcDims, ...}`.
pub use misa_wbc::{HoQp, Task, WarmStart, WbcDims};
pub use solver::{Formulation, HqpStrategy, QpSolver, WbcSolver};
pub use misa_wbc::SolveConfig;
pub use wbc::{
    solve, solve_warm, solve_warm_with_weights, WbcInputs, WbcSolution, WbcWarmStart,
    WbcWeights,
};
