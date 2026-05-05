//! Floating-base equation of motion task.
//!
//! Equality (priority 0, hard):
//!
//! ```text
//! M(q)·q̈ + h(q, q̇) − Jᵀ·f_GRF − Sᵀ·τ = 0
//! ```
//!
//! where `Sᵀ ∈ R^(nv × na)` is the actuation selection matrix:
//!
//! ```text
//! Sᵀ = [ 0_(6 × na)
//!        I_(na × na) ]
//! ```
//!
//! In our decision-variable layout `x = [q̈; f_GRF; τ]` this becomes:
//!
//! ```text
//! [ M  −Jᵀ  −Sᵀ ] · x  =  −h
//! ```

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

pub fn formulate(
    dims: WbcDims,
    mass: &DMatrix<f64>,
    nle: &DVector<f64>,
    j_contact: &DMatrix<f64>,
) -> Task {
    let n = dims.n_decision();
    debug_assert_eq!(mass.shape(), (dims.nv, dims.nv), "M must be nv × nv");
    debug_assert_eq!(nle.len(), dims.nv, "h must have length nv");
    debug_assert_eq!(
        j_contact.shape(),
        (3 * dims.nc, dims.nv),
        "j_contact must be (3·nc) × nv (linear part of stacked foot Jacobians)"
    );

    let mut a = DMatrix::zeros(dims.nv, n);
    a.view_mut((0, dims.q_offset()), (dims.nv, dims.nv))
        .copy_from(mass);
    let mut neg_jt = j_contact.transpose();
    neg_jt *= -1.0;
    a.view_mut((0, dims.f_offset()), (dims.nv, 3 * dims.nc))
        .copy_from(&neg_jt);
    // -Sᵀ at the τ columns: Sᵀ = [0_(base_dof × na); I_na]
    let base_dof = dims.nv - dims.na;
    for i in 0..dims.na {
        a[(base_dof + i, dims.tau_offset() + i)] = -1.0;
    }

    let b = -nle;
    Task::equality(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plug a chosen `(q̈, f, τ)` into the formulated `A·x − b` and
    /// verify it equals the EoM residual `M·q̈ + h − Jᵀ·f − Sᵀ·τ`.
    #[test]
    fn matches_eom_residual_definition() {
        let dims = WbcDims {
            nv: 9,
            nc: 2,
            na: 3,
        };
        let mass = DMatrix::identity(9, 9);
        let nle = DVector::from_vec((0..9).map(|i| i as f64 * 0.1).collect());
        let j = DMatrix::from_fn(6, 9, |i, j| ((i * 3 + j) as f64).sin());
        let task = formulate(dims, &mass, &nle, &j);

        let q_ddot = DVector::from_vec((0..9).map(|i| i as f64 + 1.0).collect());
        let f = DVector::from_vec((0..6).map(|i| 2.0 * i as f64).collect());
        let tau = DVector::from_vec(vec![1.0, -2.0, 3.0]);

        let mut x = DVector::zeros(dims.n_decision());
        x.view_mut((0, 0), (9, 1)).copy_from(&q_ddot);
        x.view_mut((9, 0), (6, 1)).copy_from(&f);
        x.view_mut((15, 0), (3, 1)).copy_from(&tau);

        let lhs = &task.a * &x - &task.b;
        let mut s_t = DMatrix::zeros(9, 3);
        for i in 0..3 {
            s_t[(6 + i, i)] = 1.0;
        }
        let expected = &mass * &q_ddot + &nle - j.transpose() * &f - &s_t * &tau;
        for i in 0..9 {
            assert!(
                (lhs[i] - expected[i]).abs() < 1e-12,
                "row {i}: residual mismatch ({} vs {})",
                lhs[i],
                expected[i]
            );
        }
    }
}
