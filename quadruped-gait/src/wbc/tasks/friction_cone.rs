//! Friction cone constraints for stance feet, plus zero-force
//! equality for swing feet.
//!
//! For each **stance** foot `i` (in contact with the ground), the
//! ground reaction force `f_i ∈ R³` must lie in the linearised friction
//! pyramid (5 inequalities):
//!
//! ```text
//!     −f_z ≤ −f_min                 (f_z ≥ f_min, see below)
//!  f_x − μ·f_z ≤ 0
//! −f_x − μ·f_z ≤ 0
//!  f_y − μ·f_z ≤ 0
//! −f_y − μ·f_z ≤ 0
//! ```
//!
//! For each **swing** foot `i` (not in contact), the GRF must be zero:
//!
//! ```text
//! f_i = 0    ⇒    [ 0  I_3  0 ] · x  =  0
//! ```
//!
//! `f_min` defaults to zero, which is the `legged_control` formulation this
//! mirrors: stance feet may push but not pull. That leaves the GRF allocation
//! redundant whenever three or four feet are down, and nothing in the QP then
//! prefers loading all of them -- a two-contact vertex satisfies every
//! constraint just as well, and the only thing arguing against it is a
//! low-weight regulariser toward the MPC's plan. On namiashi's Walk (duty
//! 0.75, so three feet down by construction) the measured three-foot support
//! is 0.61 with the WBC's torque applied against 0.84 with it zeroed, which
//! is what picking that vertex looks like from the outside.
//!
//! A positive `f_min` forbids it. It is a hard constraint at priority 0, so
//! it has to stay well below the real per-foot share -- at touchdown and
//! liftoff the commanded contact set disagrees with the physical one for a
//! tick or two, and demanding real force from a foot that is still in the air
//! makes the QP infeasible rather than merely wrong.
//!
//! Equality and inequality combined into a single Task at priority 0
//! (hard). Mirrors `legged_control`'s `formulateFrictionConeTask`.

use nalgebra::{DMatrix, DVector};

use super::super::{Task, WbcDims};

pub fn formulate(
    dims: WbcDims,
    contact_flag: [bool; 4],
    friction_mu: f64,
    f_min: f64,
) -> Task {
    debug_assert_eq!(
        dims.nc, 4,
        "friction_cone currently assumes 4 contact points"
    );

    let n_stance: usize = contact_flag.iter().filter(|&&b| b).count();
    let n_swing = dims.nc - n_stance;
    let n = dims.n_decision();

    // ── Equality: f_swing = 0 ──────────────────────────────────────
    let mut a = DMatrix::zeros(3 * n_swing, n);
    {
        let mut row = 0;
        for i in 0..dims.nc {
            if !contact_flag[i] {
                let col = dims.f_offset() + 3 * i;
                for k in 0..3 {
                    a[(row + k, col + k)] = 1.0;
                }
                row += 3;
            }
        }
    }
    let b = DVector::zeros(3 * n_swing);

    // ── Inequality: friction pyramid for stance feet ───────────────
    #[rustfmt::skip]
    let pyramid = DMatrix::from_row_slice(5, 3, &[
        0.0, 0.0, -1.0,
        1.0, 0.0, -friction_mu,
       -1.0, 0.0, -friction_mu,
        0.0, 1.0, -friction_mu,
        0.0,-1.0, -friction_mu,
    ]);
    let mut d = DMatrix::zeros(5 * n_stance, n);
    {
        let mut row = 0;
        for i in 0..dims.nc {
            if contact_flag[i] {
                let col = dims.f_offset() + 3 * i;
                d.view_mut((row, col), (5, 3)).copy_from(&pyramid);
                row += 5;
            }
        }
    }
    // Row 0 of each stance foot's pyramid block is `−f_z ≤ f[row]`, so a
    // right-hand side of `−f_min` reads `f_z ≥ f_min`.
    let mut f = DVector::zeros(5 * n_stance);
    if f_min > 0.0 {
        for i in 0..n_stance {
            f[5 * i] = -f_min;
        }
    }

    Task { a, b, d, f }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_swing_only_equalities() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let task = formulate(dims, [false; 4], 0.5, 0.0);
        assert_eq!(task.n_iq(), 0);
        assert_eq!(task.n_eq(), 12);
    }

    #[test]
    fn all_stance_only_pyramid() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let task = formulate(dims, [true; 4], 0.5, 0.0);
        assert_eq!(task.n_eq(), 0);
        assert_eq!(task.n_iq(), 20);
    }

    /// Pyramid row 1 picks out `f_x − μ·f_z`. With pure shear and zero
    /// normal force the constraint is violated (D·x > 0).
    #[test]
    fn pyramid_detects_shear_violation() {
        let dims = WbcDims { nv: 0, nc: 4, na: 0 };
        let task = formulate(dims, [true, false, false, false], 0.5, 0.0);
        let mut x = DVector::zeros(dims.n_decision());
        let off = dims.f_offset();
        x[off] = 1.0;
        x[off + 2] = 0.0;
        let lhs = &task.d * &x;
        assert!(lhs[1] > 0.0);
    }

    /// A stance foot carrying less than `f_min` violates row 0
    /// (`−f_z ≤ −f_min`); one carrying more satisfies it. Zero `f_min`
    /// restores the old "no pulling" behaviour exactly.
    #[test]
    fn min_normal_force_row() {
        let dims = WbcDims { nv: 0, nc: 4, na: 0 };
        let task = formulate(dims, [true, false, false, false], 0.5, 2.0);
        let off = dims.f_offset();
        let residual = |fz: f64| {
            let mut x = DVector::zeros(dims.n_decision());
            x[off + 2] = fz;
            (&task.d * &x)[0] - task.f[0]
        };
        assert!(residual(1.0) > 0.0, "1 N under a 2 N floor must violate");
        assert!(residual(3.0) < 0.0, "3 N over a 2 N floor must satisfy");

        let open = formulate(dims, [true, false, false, false], 0.5, 0.0);
        assert_eq!(open.f[0], 0.0);
    }
}
