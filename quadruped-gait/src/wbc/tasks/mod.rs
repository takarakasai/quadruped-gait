//! Concrete WBC task formulations. Each module encodes one row-block
//! of the HoQp's inputs in terms of:
//!
//! ```text
//! x = [ q̈   |   f_GRF   |   τ ]
//! ```
//!
//! Tasks at the same priority level are combined via `+` before being
//! handed to [`super::HoQp`].

pub mod floating_base_eom;
pub mod torque_limits;
pub mod friction_cone;
pub mod no_contact_motion;
pub mod base_accel;
pub mod swing_leg;
pub mod contact_force;
