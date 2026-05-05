//! No-contact-motion equality for stance feet.
//!
//! A foot in contact with the ground must have **zero world-frame
//! linear acceleration**. From the kinematic identity
//!
//! ```text
//! a_foot_world = J_foot · q̈ + J̇_foot · q̇
//! ```
//!
//! the constraint `a_foot_world = 0` becomes the equality
//!
//! ```text
//! J_foot · q̈ = − J̇_foot · q̇
//! ```
//!
//! formulated row-by-row for each **stance** foot only. Swing feet are
//! left out (their motion is shaped by [`super::swing_leg`] instead).
//!
//! Layered into priority 0 (hard equality) so any non-zero stance-foot
//! acceleration is treated as a constraint violation rather than a
//! tracking error.

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

pub fn formulate(
    dims: WbcDims,
    j_contact: &DMatrix<f64>,
    dj_v: &DVector<f64>,
    contact_flag: [bool; 4],
) -> Task {
    debug_assert_eq!(j_contact.shape(), (3 * dims.nc, dims.nv));
    debug_assert_eq!(dj_v.len(), 3 * dims.nc);

    let n_stance: usize = contact_flag.iter().filter(|&&b| b).count();
    let n = dims.n_decision();

    let mut a = DMatrix::zeros(3 * n_stance, n);
    let mut b = DVector::zeros(3 * n_stance);

    let mut row = 0;
    for i in 0..dims.nc {
        if !contact_flag[i] {
            continue;
        }
        // Copy 3 rows of the foot's linear Jacobian into the q̈ block.
        let j_block = j_contact.view((3 * i, 0), (3, dims.nv));
        a.view_mut((row, dims.q_offset()), (3, dims.nv))
            .copy_from(&j_block);
        for k in 0..3 {
            b[row + k] = -dj_v[3 * i + k];
        }
        row += 3;
    }

    Task::equality(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_when_all_swing() {
        let dims = WbcDims { nv: 6, nc: 4, na: 0 };
        let j = DMatrix::identity(12, 6);
        let djv = DVector::zeros(12);
        let task = formulate(dims, &j, &djv, [false; 4]);
        assert_eq!(task.n_eq(), 0);
    }

    /// With one stance foot, the equality A·x = b should reproduce
    /// `J_foot · q̈ = −dJ·v` row-for-row.
    #[test]
    fn one_stance_picks_correct_rows() {
        let dims = WbcDims { nv: 6, nc: 4, na: 0 };
        // Block-diagonal-ish J: foot 2's rows are unique so we can spot them.
        let mut j = DMatrix::zeros(12, 6);
        for r in 6..9 {
            for c in 0..6 {
                j[(r, c)] = (r * 10 + c) as f64;
            }
        }
        let mut djv = DVector::zeros(12);
        djv[6] = 1.0;
        djv[7] = 2.0;
        djv[8] = 3.0;

        let task = formulate(dims, &j, &djv, [false, false, true, false]);
        assert_eq!(task.n_eq(), 3);
        // Row 0 of A = row 6 of j_contact.
        for c in 0..6 {
            assert_eq!(task.a[(0, c)], j[(6, c)]);
        }
        assert_eq!(task.b[0], -1.0);
        assert_eq!(task.b[1], -2.0);
        assert_eq!(task.b[2], -3.0);
    }
}
