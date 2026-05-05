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
