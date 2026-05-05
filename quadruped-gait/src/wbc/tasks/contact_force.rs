//! Contact-force regularisation task.
//!
//! Soft equality (priority 2, lowest):
//!
//! ```text
//! f_GRF = f_MPC
//! ```
//!
//! Picks the entire `f_GRF` block of the decision vector and ties it to
//! the SRBD MPC's predicted ground reaction forces. Acts as a
//! regulariser: among all `(q̈, f, τ)` triples that satisfy the higher-
//! priority hard constraints AND the priority-1 tracking tasks, prefer
//! the one whose contact forces match the MPC plan.

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

/// `f_grf_des` is the stacked MPC-predicted GRF (3·nc) in world frame.
pub fn formulate(dims: WbcDims, f_grf_des: &DVector<f64>) -> Task {
    debug_assert_eq!(
        f_grf_des.len(),
        3 * dims.nc,
        "f_grf_des must have 3·nc entries"
    );

    let mut a = DMatrix::zeros(3 * dims.nc, dims.n_decision());
    for i in 0..(3 * dims.nc) {
        a[(i, dims.f_offset() + i)] = 1.0;
    }
    Task::equality(a, f_grf_des.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_f_grf_block_only() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let f_des = DVector::from_fn(12, |i, _| (i + 1) as f64);
        let task = formulate(dims, &f_des);
        assert_eq!(task.n_eq(), 12);
        // Verify A·x for x = e_(f_offset+k) gives e_k.
        for k in 0..12 {
            let mut x = DVector::zeros(dims.n_decision());
            x[dims.f_offset() + k] = 1.0;
            let lhs = &task.a * &x;
            for i in 0..12 {
                let expected = if i == k { 1.0 } else { 0.0 };
                assert_eq!(lhs[i], expected);
            }
        }
        // b carries the MPC reference verbatim.
        for i in 0..12 {
            assert_eq!(task.b[i], (i + 1) as f64);
        }
    }
}
