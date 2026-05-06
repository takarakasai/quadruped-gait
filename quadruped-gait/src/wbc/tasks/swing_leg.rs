//! Joint-space swing-leg PD tracking task.
//!
//! Soft equality (priority 1, only for actuators belonging to legs
//! that are **not** in contact):
//!
//! ```text
//! q̈[6 + a]  =  q̈_des_swing[a]
//! ```
//!
//! where `q̈_des_swing[a] = K_p · (q*[a] − q[a]) + K_d · (q̇*[a] − q̇[a])`
//! is the joint-space PD computed by the host before calling the WBC,
//! using the **same `q*` that Position-PD tracks** (the gait
//! controller's IK output of the swing trajectory).
//!
//! ## Why joint-space rather than Cartesian
//!
//! `legged_control`'s WBC sources both the swing-leg target and the
//! Position-PD reference from OCS2's predicted joint state, so the
//! two paths are intrinsically consistent. articara's SRBD MPC only
//! emits base + GRF — no joint-level reference — and the gait
//! controller produces a Cartesian foot trajectory + an IK-derived
//! `q*`. Computing the WBC swing target from the Cartesian path
//! (forward kinematics + a Cartesian PD) introduces a parallel
//! representation that doesn't quite agree with what Position-PD is
//! tracking; the resulting torque conflict was empirically
//! observed to drag the body **backward** under a forward command
//! (see [`tests/wbc_walk.rs`]'s `wbc_forward_command_advances_body`).
//!
//! Tracking `q*` directly in joint space makes the WBC's swing
//! reference identical to Position-PD's reference, so the two
//! cooperate instead of fighting.
//!
//! Stance feet are skipped (the stance constraint lives in
//! `no_contact_motion` at priority 0).

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

/// Build the joint-space swing-leg tracking task.
///
/// `swing_q_ddot_des` has length `na` (one entry per actuator). Only
/// rows where `swing_actuator_flag[i]` is true contribute — others
/// are skipped. The flag is true for actuators whose leg is currently
/// in **swing**; the host sets this from
/// [`crate::ControllerOutput`]'s per-leg `phase.is_stance` together
/// with the actuator-to-leg mapping.
pub fn formulate(
    dims: WbcDims,
    swing_q_ddot_des: &DVector<f64>,
    swing_actuator_flag: &[bool],
) -> Task {
    debug_assert_eq!(swing_q_ddot_des.len(), dims.na, "length must be na");
    debug_assert_eq!(swing_actuator_flag.len(), dims.na, "length must be na");

    let n_swing = swing_actuator_flag.iter().filter(|&&b| b).count();
    let n = dims.n_decision();
    let mut a = DMatrix::zeros(n_swing, n);
    let mut b = DVector::zeros(n_swing);
    let mut row = 0;
    for i in 0..dims.na {
        if !swing_actuator_flag[i] {
            continue;
        }
        // q̈ for actuated joint i lives at decision-vector index
        // `q_offset + 6 + i` (the 6 base-DoF entries come first).
        a[(row, dims.q_offset() + 6 + i)] = 1.0;
        b[row] = swing_q_ddot_des[i];
        row += 1;
    }
    Task::equality(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_when_all_stance() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let q_dd_des = DVector::zeros(12);
        let task = formulate(dims, &q_dd_des, &[false; 12]);
        assert_eq!(task.n_eq(), 0);
    }

    /// One actuator marked swing → one equality row picking that
    /// actuator's q̈ = q_dd_des[i] (with the right `q_offset + 6 + i`
    /// column index).
    #[test]
    fn one_swing_picks_actuator_row() {
        let dims = WbcDims { nv: 9, nc: 4, na: 3 };
        let q_dd_des = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let task = formulate(dims, &q_dd_des, &[false, true, false]);
        assert_eq!(task.n_eq(), 1);
        // The single row picks q̈[q_offset + 6 + 1] (actuator 1).
        for j in 0..dims.n_decision() {
            let expected = if j == dims.q_offset() + 6 + 1 { 1.0 } else { 0.0 };
            assert_eq!(task.a[(0, j)], expected);
        }
        assert_eq!(task.b[0], 2.0);
    }

    /// Multiple swing actuators stack their rows in actuator order.
    #[test]
    fn multiple_swing_actuators_stack() {
        let dims = WbcDims { nv: 9, nc: 4, na: 3 };
        let q_dd_des = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let task = formulate(dims, &q_dd_des, &[true, false, true]);
        assert_eq!(task.n_eq(), 2);
        // Row 0 picks actuator 0 → b = 1.0
        assert_eq!(task.b[0], 1.0);
        assert_eq!(task.a[(0, dims.q_offset() + 6 + 0)], 1.0);
        // Row 1 picks actuator 2 → b = 3.0
        assert_eq!(task.b[1], 3.0);
        assert_eq!(task.a[(1, dims.q_offset() + 6 + 2)], 1.0);
    }
}
