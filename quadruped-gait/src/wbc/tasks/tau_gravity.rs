//! τ regularisation toward static gravity-compensation torques.
//!
//! Soft equality (lowest priority, run **after** contact_force):
//!
//! ```text
//! τ = τ_grav
//! ```
//!
//! where `τ_grav = τ_grav(q)` is the joint torque that statically
//! supports the robot under gravity at the current configuration with
//! `q̇ = 0`, `q̈ = 0`. Computed by the host via
//! `misarta::rnea::compute_gravity` and projected to actuator rows.
//!
//! Without this anchor, the WBC's QP at static stand has multiple
//! feasible `(q̈, f, τ)` triples — including the pathological
//! `(0, m·g/4·e_z per foot, τ ≈ 0)` where contact forces alone
//! balance gravity and joint torque is irrelevant. clarabel's
//! interior point picks among these arbitrarily, so τ jumps tick-to-
//! tick (1.6 → 13.6 N·m observed at namiashi static stand). With
//! `tau_gravity` as a soft target at priority 3, the optimiser
//! prefers solutions where joints actually produce the
//! gravity-comp torques — the legs hold up the body even when GRFs
//! could nominally do it alone.

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

/// Build the τ ≈ τ_grav equality task. Use `Task::weight()` from
/// the caller to bias against same-priority competitors (e.g.
/// `contact_force`).
pub fn formulate(dims: WbcDims, tau_gravity: &DVector<f64>) -> Task {
    debug_assert_eq!(
        tau_gravity.len(),
        dims.na,
        "tau_gravity length must match na"
    );
    let mut a = DMatrix::zeros(dims.na, dims.n_decision());
    for i in 0..dims.na {
        a[(i, dims.tau_offset() + i)] = 1.0;
    }
    let b = tau_gravity.clone();
    Task::equality(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `A` picks out only the τ block of the decision vector and
    /// `b` carries the gravity-comp reference verbatim.
    #[test]
    fn picks_tau_block_and_carries_reference() {
        let dims = WbcDims { nv: 6, nc: 1, na: 2 };
        let tau_g = DVector::from_vec(vec![1.5, -2.5]);
        let task = formulate(dims, &tau_g);
        assert_eq!(task.n_eq(), 2);
        // Row i of A picks decision[tau_offset + i] = τ_i.
        for i in 0..2 {
            for j in 0..dims.n_decision() {
                let expected = if j == dims.tau_offset() + i { 1.0 } else { 0.0 };
                assert_eq!(task.a[(i, j)], expected);
            }
        }
        assert_eq!(task.b[0], 1.5);
        assert_eq!(task.b[1], -2.5);
    }
}
