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

/// Friction cone with a **continuous contact weight** per foot instead of a
/// boolean flag.
///
/// [`formulate`] switches a foot between "force must be exactly zero"
/// (a hard equality) and "force may be anywhere in the cone". The rank of
/// the priority-0 block therefore jumps by 3 on the tick a foot changes
/// state, and a hierarchical least-squares solver can return a point that
/// satisfies none of its blocks on exactly that tick. Measured on a 53 kg
/// quadruped in MuJoCo: total vertical reaction 0.1 N against a 523 N
/// machine while the solved base angular acceleration was 57 rad/s², once
/// per stance swap.
///
/// Weighting removes the switch. Every foot gets the same six inequalities
/// every tick and only the bounds move:
///
/// ```text
///     −f_z ≤ −w·f_min           (f_z ≥ w·f_min)
///      f_z ≤  w·f_max
///  ±f_x − μ·f_z ≤ 0
///  ±f_y − μ·f_z ≤ 0
/// ```
///
/// **At `w = 0` this is the swing equality.** `0 ≤ f_z ≤ 0` pins the normal
/// force, and the pyramid then pins the tangential one, so `f = 0` follows
/// without a separate equality block. At `w = 1` it is [`formulate`] plus an
/// upper bound. In between the foot may carry a fraction of the load, which
/// is what a foot that is leaving or arriving is physically doing.
///
/// `f_max` is the per-foot vertical cap at full weight; pass something like
/// twice the machine's weight. It must be finite -- the cap is what makes
/// the ramp mean anything.
pub fn formulate_weighted(
    dims: WbcDims,
    contact_weight: [f64; 4],
    friction_mu: f64,
    f_min: f64,
    f_max: f64,
) -> Task {
    debug_assert_eq!(
        dims.nc, 4,
        "friction_cone currently assumes 4 contact points"
    );
    debug_assert!(f_max.is_finite() && f_max > 0.0, "f_max must be finite");

    let n = dims.n_decision();
    // Six rows per foot, always. **The row count must not depend on the
    // contact state** -- that is the whole point.
    let mut d = DMatrix::zeros(6 * dims.nc, n);
    let mut f = DVector::zeros(6 * dims.nc);
    for i in 0..dims.nc {
        let w = contact_weight[i].clamp(0.0, 1.0);
        let col = dims.f_offset() + 3 * i;
        let row = 6 * i;
        // f_z ≥ w·f_min
        d[(row, col + 2)] = -1.0;
        f[row] = -w * f_min;
        // f_z ≤ w·f_max
        d[(row + 1, col + 2)] = 1.0;
        f[row + 1] = w * f_max;
        // ±f_x − μ·f_z ≤ 0, ±f_y − μ·f_z ≤ 0
        for (k, axis) in [0usize, 1].into_iter().enumerate() {
            let r = row + 2 + 2 * k;
            d[(r, col + axis)] = 1.0;
            d[(r, col + 2)] = -friction_mu;
            d[(r + 1, col + axis)] = -1.0;
            d[(r + 1, col + 2)] = -friction_mu;
        }
    }

    Task {
        a: DMatrix::zeros(0, n),
        b: DVector::zeros(0),
        d,
        f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The block keeps its shape whatever the contact state is.** That is
    /// what stops the rank of priority 0 from jumping mid-stride.
    #[test]
    fn the_weighted_block_has_a_constant_shape() {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        for w in [[0.0; 4], [1.0; 4], [0.0, 0.3, 0.7, 1.0]] {
            let task = formulate_weighted(dims, w, 0.5, 0.0, 1000.0);
            assert_eq!(task.n_eq(), 0);
            assert_eq!(task.n_iq(), 24);
        }
    }

    /// **Weight zero is the swing equality.** `0 ≤ f_z ≤ 0` pins the normal
    /// force and the pyramid then pins the tangential one, so any non-zero
    /// force violates something.
    #[test]
    fn zero_weight_pins_the_foot_force_to_zero() {
        let dims = WbcDims { nv: 0, nc: 4, na: 0 };
        let task = formulate_weighted(dims, [0.0; 4], 0.5, 0.0, 1000.0);
        let off = dims.f_offset();
        for (axis, name) in [(0, "f_x"), (1, "f_y"), (2, "f_z")] {
            for sign in [1.0, -1.0] {
                let mut x = DVector::zeros(dims.n_decision());
                x[off + axis] = sign * 10.0;
                let lhs = &task.d * &x;
                assert!(
                    (0..task.n_iq()).any(|r| lhs[r] > task.f[r] + 1e-9),
                    "{name} = {sign}·10 should violate something at w = 0"
                );
            }
        }
        // Zero force itself is feasible.
        let x = DVector::zeros(dims.n_decision());
        let lhs = &task.d * &x;
        assert!((0..task.n_iq()).all(|r| lhs[r] <= task.f[r] + 1e-9));
    }

    /// The cap scales with the weight, so a foot that is half in contact may
    /// carry half the load.
    #[test]
    fn the_normal_force_cap_follows_the_weight() {
        let dims = WbcDims { nv: 0, nc: 4, na: 0 };
        let task = formulate_weighted(dims, [0.5, 0.0, 0.0, 0.0], 0.5, 0.0, 800.0);
        let off = dims.f_offset();
        let feasible = |fz: f64| {
            let mut x = DVector::zeros(dims.n_decision());
            x[off + 2] = fz;
            let lhs = &task.d * &x;
            (0..task.n_iq()).all(|r| lhs[r] <= task.f[r] + 1e-9)
        };
        assert!(feasible(399.0));
        assert!(!feasible(401.0));
    }

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
