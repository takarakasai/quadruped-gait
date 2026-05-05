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

pub mod dims;
pub mod ho_qp;
pub mod task;
pub mod tasks;
mod wbc;

pub use dims::WbcDims;
pub use ho_qp::HoQp;
pub use task::Task;
pub use wbc::{solve, WbcInputs, WbcSolution};
