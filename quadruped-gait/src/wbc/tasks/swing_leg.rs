//! Swing-leg PD tracking task.
//!
//! Soft equality (priority 1, only for legs **not** in contact):
//!
//! ```text
//! a_foot_world = a_swing_des
//! ```
//!
//! where the desired swing-foot acceleration comes from a Cartesian
//! PD on the planned swing trajectory (computed by the host before
//! calling the WBC):
//!
//! ```text
//! a_swing_des = K_p · (p_des − p_meas)  +  K_d · (v_des − v_meas)
//! ```
//!
//! Using the same kinematic identity as
//! [`super::no_contact_motion`] (`a_foot = J·q̈ + J̇·q̇`):
//!
//! ```text
//! J_foot · q̈ = a_swing_des − J̇·q̇
//! ```
//!
//! Stance feet are skipped (their stance constraint is in
//! `no_contact_motion` at priority 0).

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

/// `a_swing_des` is a stacked `(3·nc)`-vector: per-foot Cartesian
/// acceleration target in the world frame. Entries for stance feet
/// are ignored (those rows are skipped in the output Task).
pub fn formulate(
    dims: WbcDims,
    j_contact: &DMatrix<f64>,
    dj_v: &DVector<f64>,
    a_swing_des: &DVector<f64>,
    contact_flag: [bool; 4],
) -> Task {
    debug_assert_eq!(j_contact.shape(), (3 * dims.nc, dims.nv));
    debug_assert_eq!(dj_v.len(), 3 * dims.nc);
    debug_assert_eq!(a_swing_des.len(), 3 * dims.nc);

    let n_swing = dims.nc - contact_flag.iter().filter(|&&b| b).count();
    let n = dims.n_decision();

    let mut a = DMatrix::zeros(3 * n_swing, n);
    let mut b = DVector::zeros(3 * n_swing);
    let mut row = 0;
    for i in 0..dims.nc {
        if contact_flag[i] {
            continue;
        }
        let j_block = j_contact.view((3 * i, 0), (3, dims.nv));
        a.view_mut((row, dims.q_offset()), (3, dims.nv))
            .copy_from(&j_block);
        for k in 0..3 {
            b[row + k] = a_swing_des[3 * i + k] - dj_v[3 * i + k];
        }
        row += 3;
    }

    Task::equality(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_when_all_stance() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let j = DMatrix::zeros(12, 18);
        let djv = DVector::zeros(12);
        let a_des = DVector::zeros(12);
        let task = formulate(dims, &j, &djv, &a_des, [true; 4]);
        assert_eq!(task.n_eq(), 0);
    }

    #[test]
    fn one_swing_picks_target_minus_dj_v() {
        let dims = WbcDims { nv: 6, nc: 4, na: 0 };
        let mut j = DMatrix::zeros(12, 6);
        for r in 3..6 {
            for c in 0..6 {
                j[(r, c)] = ((r + 1) * 100 + c) as f64;
            }
        }
        let mut djv = DVector::zeros(12);
        djv[3] = 0.5;
        djv[4] = 0.6;
        djv[5] = 0.7;
        let mut a_des = DVector::zeros(12);
        a_des[3] = 1.0;
        a_des[4] = 2.0;
        a_des[5] = 3.0;
        let task = formulate(dims, &j, &djv, &a_des, [true, false, true, true]);
        assert_eq!(task.n_eq(), 3);
        // b = a_des - dJ·v
        assert_eq!(task.b[0], 0.5);
        assert_eq!(task.b[1], 1.4);
        assert_eq!(task.b[2], 2.3);
    }
}
