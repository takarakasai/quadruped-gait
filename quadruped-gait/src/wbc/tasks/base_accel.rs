//! Base acceleration tracking task.
//!
//! Soft equality (priority 1):
//!
//! ```text
//! q̈[0..6] = a_base_des
//! ```
//!
//! The first 6 entries of the generalised acceleration drive the
//! floating-base 6-DoF (linear + angular). The reference comes from
//! the SRBD MPC, which provides the base acceleration that produces
//! the commanded body velocity over the next horizon step.
//!
//! In our decision layout `x = [q̈; f_GRF; τ]` the task selects the
//! first 6 rows of `q̈`:
//!
//! ```text
//! [ I_6  0_(6 × 3·nc)  0_(6 × na) ] · x  =  a_base_des
//! ```

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

/// Build the base-acceleration equality task.
///
/// `a_base_des` is laid out `[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]`
/// in the world frame.
pub fn formulate(dims: WbcDims, a_base_des: &DVector<f64>) -> Task {
    debug_assert_eq!(a_base_des.len(), 6, "base accel reference must have 6 entries");
    debug_assert!(dims.nv >= 6, "base accel task assumes a floating base (nv ≥ 6)");

    let mut a = DMatrix::zeros(6, dims.n_decision());
    for i in 0..6 {
        a[(i, dims.q_offset() + i)] = 1.0;
    }
    Task::equality(a, a_base_des.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_first_six_rows_of_q_ddot() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let a_des = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let task = formulate(dims, &a_des);
        assert_eq!(task.n_eq(), 6);
        // A picks columns [0..6] → identity on q̈[0..6], zero elsewhere.
        for i in 0..6 {
            for j in 0..dims.n_decision() {
                let expected = if j == i { 1.0 } else { 0.0 };
                assert_eq!(task.a[(i, j)], expected);
            }
        }
        for i in 0..6 {
            assert_eq!(task.b[i], a_des[i]);
        }
    }
}
