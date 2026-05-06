//! Linear constraint container used as the elementary unit of every
//! [`super::HoQp`] priority level.
//!
//! A `Task` packages **soft equalities** `A·x = b` (will be tracked in
//! the least-squares sense and pushed into the cost) together with
//! **hard inequalities** `D·x ≤ f` (must be feasible). Either side is
//! optional — pure-equality and pure-inequality tasks are common.
//!
//! Tasks at the same priority level are combined with `+` (concatenate
//! rows). The HoQp builder consumes a chain of these to assemble each
//! priority level's internal QP.

use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct Task {
    /// Equality side: `A·x = b` (treated as soft / least-squares cost
    /// at this priority level). `A` has shape `(neq, n_decision)`.
    pub a: DMatrix<f64>,
    pub b: DVector<f64>,
    /// Inequality side: `D·x ≤ f` (hard constraint at this priority
    /// level — slack variables push violations into the next-level
    /// minimisation). `D` has shape `(niq, n_decision)`.
    pub d: DMatrix<f64>,
    pub f: DVector<f64>,
}

impl Task {
    /// Empty task on a problem with `n_decision` decision variables.
    /// Used as a starting point that subsequent tasks can be added to.
    pub fn empty(n_decision: usize) -> Self {
        Self {
            a: DMatrix::zeros(0, n_decision),
            b: DVector::zeros(0),
            d: DMatrix::zeros(0, n_decision),
            f: DVector::zeros(0),
        }
    }

    /// Equality-only task `A·x = b`. Inequality side is empty.
    pub fn equality(a: DMatrix<f64>, b: DVector<f64>) -> Self {
        let n = a.ncols();
        Self {
            a,
            b,
            d: DMatrix::zeros(0, n),
            f: DVector::zeros(0),
        }
    }

    /// Inequality-only task `D·x ≤ f`. Equality side is empty.
    pub fn inequality(d: DMatrix<f64>, f: DVector<f64>) -> Self {
        let n = d.ncols();
        Self {
            a: DMatrix::zeros(0, n),
            b: DVector::zeros(0),
            d,
            f,
        }
    }

    /// Number of decision variables this task is defined over.
    /// Reads from `A` if non-empty, falls back to `D`.
    pub fn n_decision(&self) -> usize {
        if self.a.ncols() > 0 {
            self.a.ncols()
        } else {
            self.d.ncols()
        }
    }

    pub fn n_eq(&self) -> usize {
        self.a.nrows()
    }

    pub fn n_iq(&self) -> usize {
        self.d.nrows()
    }

    /// Scale the equality residual at this task by `weight`. The HoQp
    /// inner cost ½‖A·x − b‖² becomes ½ weight²·‖A·x − b‖², making
    /// this task's residual `weight²` more expensive than an
    /// unweighted one stacked at the same priority level.
    ///
    /// Use to bias which task "wins" inside a single priority level
    /// when several conflict (e.g. base_accel vs swing_leg). Equivalent
    /// to multiplying both `A` and `b` by `√weight` so the LSQ scales
    /// quadratically.
    ///
    /// `weight` must be ≥ 0. Inequalities are not scaled — they are
    /// already feasibility-only, and scaling their slack would just
    /// renormalise the slack-cost block to no effect.
    pub fn weight(mut self, weight: f64) -> Self {
        assert!(weight >= 0.0, "Task::weight: weight must be ≥ 0");
        let s = weight.sqrt();
        if (s - 1.0).abs() > 1e-15 {
            self.a *= s;
            self.b *= s;
        }
        self
    }
}

impl std::ops::Add for Task {
    type Output = Task;
    /// Stack two tasks at the same priority level (concatenate rows).
    /// Both tasks must reference the same decision variable space.
    fn add(self, rhs: Task) -> Task {
        let n = self.n_decision().max(rhs.n_decision());
        debug_assert!(
            (self.a.ncols() == 0 || self.a.ncols() == n)
                && (self.d.ncols() == 0 || self.d.ncols() == n)
                && (rhs.a.ncols() == 0 || rhs.a.ncols() == n)
                && (rhs.d.ncols() == 0 || rhs.d.ncols() == n),
            "Task::add: decision variable size mismatch"
        );
        Task {
            a: stack_rows(&self.a, &rhs.a, n),
            b: stack_vec(&self.b, &rhs.b),
            d: stack_rows(&self.d, &rhs.d, n),
            f: stack_vec(&self.f, &rhs.f),
        }
    }
}

fn stack_rows(m1: &DMatrix<f64>, m2: &DMatrix<f64>, ncols: usize) -> DMatrix<f64> {
    if m1.nrows() == 0 {
        return if m2.nrows() == 0 {
            DMatrix::zeros(0, ncols)
        } else {
            m2.clone()
        };
    }
    if m2.nrows() == 0 {
        return m1.clone();
    }
    let mut out = DMatrix::zeros(m1.nrows() + m2.nrows(), ncols);
    out.view_mut((0, 0), (m1.nrows(), ncols)).copy_from(m1);
    out.view_mut((m1.nrows(), 0), (m2.nrows(), ncols)).copy_from(m2);
    out
}

fn stack_vec(v1: &DVector<f64>, v2: &DVector<f64>) -> DVector<f64> {
    if v1.is_empty() {
        return v2.clone();
    }
    if v2.is_empty() {
        return v1.clone();
    }
    let mut out = DVector::zeros(v1.len() + v2.len());
    out.view_mut((0, 0), (v1.len(), 1)).copy_from(v1);
    out.view_mut((v1.len(), 0), (v2.len(), 1)).copy_from(v2);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_task_has_correct_shape() {
        let t = Task::empty(5);
        assert_eq!(t.n_decision(), 5);
        assert_eq!(t.n_eq(), 0);
        assert_eq!(t.n_iq(), 0);
    }

    #[test]
    fn add_concatenates_rows() {
        let t1 = Task::equality(
            DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            DVector::from_vec(vec![1.0, 2.0]),
        );
        let t2 = Task::inequality(
            DMatrix::from_row_slice(1, 3, &[0.0, 0.0, 1.0]),
            DVector::from_vec(vec![5.0]),
        );
        let s = t1 + t2;
        assert_eq!(s.n_eq(), 2);
        assert_eq!(s.n_iq(), 1);
        assert_eq!(s.n_decision(), 3);
    }

    /// `weight(w)` scales `A` and `b` by `√w` so the LSQ cost
    /// ½‖A·x − b‖² ends up multiplied by `w`.
    #[test]
    fn weight_scales_a_and_b_by_sqrt() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DVector::from_vec(vec![5.0, 6.0]);
        let t = Task::equality(a.clone(), b.clone()).weight(4.0);
        // weight 4 → scale = 2.
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!((t.a[(i, j)] - 2.0 * a[(i, j)]).abs() < 1e-15);
            }
            assert!((t.b[i] - 2.0 * b[i]).abs() < 1e-15);
        }
    }

    /// `weight(1.0)` must be a no-op so call sites that don't want
    /// scaling can chain it harmlessly.
    #[test]
    fn weight_one_is_a_noop() {
        let a = DMatrix::from_row_slice(1, 2, &[1.5, -2.5]);
        let b = DVector::from_vec(vec![3.5]);
        let t = Task::equality(a.clone(), b.clone()).weight(1.0);
        for j in 0..2 {
            assert!((t.a[(0, j)] - a[(0, j)]).abs() < 1e-15);
        }
        assert!((t.b[0] - b[0]).abs() < 1e-15);
    }

    /// Inequality side is left untouched by `weight()`.
    #[test]
    fn weight_does_not_scale_inequalities() {
        let d = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let f = DVector::from_vec(vec![10.0]);
        let t = Task::inequality(d.clone(), f.clone()).weight(9.0);
        // D, f unchanged.
        assert!((t.d[(0, 0)] - 1.0).abs() < 1e-15);
        assert!((t.d[(0, 1)] - 1.0).abs() < 1e-15);
        assert!((t.f[0] - 10.0).abs() < 1e-15);
    }

    #[test]
    fn add_two_equalities_stacks_them() {
        let t1 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DVector::from_vec(vec![1.0]),
        );
        let t2 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[0.0, 1.0]),
            DVector::from_vec(vec![2.0]),
        );
        let s = t1 + t2;
        assert_eq!(s.n_eq(), 2);
        assert_eq!(s.b[0], 1.0);
        assert_eq!(s.b[1], 2.0);
    }
}
