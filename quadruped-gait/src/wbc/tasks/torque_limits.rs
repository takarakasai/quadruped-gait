//! Per-actuator torque limits.
//!
//! Inequality (priority 0, hard):
//!
//! ```text
//!  τ ≤ +τ_max          ⇒   [ 0  0  +I_na ] · x  ≤  τ_max
//! −τ ≤ +τ_max          ⇒   [ 0  0  −I_na ] · x  ≤  τ_max
//! ```
//!
//! Layered into `D·x ≤ f`. Replaces (and supersedes) the
//! [`crate::config::JointData::effort`] hard-clip in `MujocoSim` —
//! when WBC is active the torque request never exceeds `τ_max` in the
//! first place.

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

pub fn formulate(dims: WbcDims, torque_max: &DVector<f64>) -> Task {
    debug_assert_eq!(
        torque_max.len(),
        dims.na,
        "torque_max length must match na"
    );

    let n = dims.n_decision();
    let mut d = DMatrix::zeros(2 * dims.na, n);
    let mut f = DVector::zeros(2 * dims.na);

    // +τ ≤ τ_max
    for i in 0..dims.na {
        d[(i, dims.tau_offset() + i)] = 1.0;
        f[i] = torque_max[i];
    }
    // −τ ≤ τ_max
    for i in 0..dims.na {
        d[(dims.na + i, dims.tau_offset() + i)] = -1.0;
        f[dims.na + i] = torque_max[i];
    }

    Task::inequality(d, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inequality picks out `τ` only — `q̈` and `f_GRF` columns must be
    /// zero, and the row count is `2·na`.
    #[test]
    fn picks_only_tau_block_with_correct_bounds() {
        let dims = WbcDims {
            nv: 6,
            nc: 1,
            na: 2,
        };
        let limits = DVector::from_vec(vec![10.0, 20.0]);
        let task = formulate(dims, &limits);
        assert_eq!(task.n_iq(), 4);
        assert_eq!(task.n_decision(), dims.n_decision());

        // Row 0: +τ[0] ≤ 10
        let row = task.d.row(0);
        for j in 0..dims.tau_offset() {
            assert_eq!(row[j], 0.0, "non-zero outside τ block at col {j}");
        }
        assert_eq!(row[dims.tau_offset()], 1.0);
        assert_eq!(row[dims.tau_offset() + 1], 0.0);
        assert_eq!(task.f[0], 10.0);

        // Row 3: -τ[1] ≤ 20
        let row = task.d.row(3);
        assert_eq!(row[dims.tau_offset()], 0.0);
        assert_eq!(row[dims.tau_offset() + 1], -1.0);
        assert_eq!(task.f[3], 20.0);
    }
}
