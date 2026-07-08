//! Hierarchical Quadratic Program solver.
//!
//! Implementation follows [Kim et al. 2014](https://doi.org/10.1109/TRO.2013.2293057)
//! "An Optimization Approach to Whole-Body Manipulation". Each
//! [`super::Task`] is solved at its priority level **inside the null
//! space** of all higher-priority tasks' equality constraints, so the
//! higher-priority solution is preserved exactly while the current
//! task is satisfied as best as possible.
//!
//! Per-level QP (decision variables `[v; w]`):
//!
//! ```text
//! min_{v, w}   ½ ‖A_k Z_{k−1} v − (b_k − A_k x_{k−1})‖²  +  ½ ‖w‖²
//!
//! s.t.   D_k Z_{k−1} v − w  ≤  f_k − D_k x_{k−1}        (current task ineq)
//!        D_{<k} Z_{k−1} v   ≤  f_{<k} − D_{<k} x_{k−1} + w_{<k}  (kept from above)
//!        −w                 ≤  0                       (slack ≥ 0)
//! ```
//!
//! After each level we update:
//! ```text
//! x_k = x_{k−1} + Z_{k−1} v_k*
//! Z_k = Z_{k−1} · null(A_k Z_{k−1})
//! ```
//!
//! Use is one-shot: build the chain `HoQp::new(task_0, None) → HoQp::new(task_1, Some(prev))
//! → HoQp::new(task_2, Some(prev2))`, then call `solution()` on the
//! deepest level. Mirrors `legged_control`'s API for line-by-line
//! diff-ability.

use nalgebra::{DMatrix, DVector};

use misarta::qp::{solve_qp, QpConfig, QpSolver, QpStatus};

use super::Task;

/// Per-level state retained for the next-level builder. Public only
/// through [`HoQp::previous`] so subclasses / debug tooling can peek;
/// callers normally just use [`HoQp::solution`].
#[derive(Clone, Debug)]
struct LevelState {
    /// Cumulative solution after this level: `x_k = x_{k−1} + Z_{k−1} v_k*`.
    x: DVector<f64>,
    /// Null-space basis of all equality constraints up to this level
    /// (inclusive). Columns span the directions in which the next
    /// level may move without disturbing the current solution.
    z: DMatrix<f64>,
    /// All inequality rows accumulated up to and including this level
    /// (`D_{≤k}`, `f_{≤k}`).
    stacked_d: DMatrix<f64>,
    stacked_f: DVector<f64>,
    /// Slack values solved at this level + everything above. The next
    /// level's QP relaxes higher inequalities by exactly these slacks
    /// so it doesn't fight the prior solution.
    stacked_slack: DVector<f64>,
}

#[derive(Clone, Debug)]
pub struct HoQp {
    state: LevelState,
    /// The raw inner-QP decision vector `y = [v; w]` solved at this
    /// level. Cached so the host can persist it between ticks and pass
    /// it back as the next tick's warm-start anchor (see
    /// [`WarmStart`]). Length = `n_v + n_slack` for this level.
    y_solution: DVector<f64>,
}

/// Warm-start hint for [`HoQp::new_with_higher_warm`].
///
/// `x_prev` is a previous tick's **full decision-space** solution
/// (length `n_decision_total`, the same as the current level's
/// final `state.x`). The inner QP at this level is augmented with
/// `(prox_weight / 2) · ‖v − v_target‖²` where
/// `v_target = prev.zᵀ · (x_prev − prev.x)` — i.e. the projection
/// of `(x_prev − prev.x)` into the current level's null-space basis.
///
/// Why project at all: at levels k>0 the inner variable `v` lives in
/// the basis `prev.z`, which is rebuilt every tick from a `q`-dependent
/// equality matrix. A naive per-level `y_prev` cache would compare
/// coordinates across DIFFERENT bases — meaningless and observed
/// empirically to AMPLIFY jitter rather than damp it. Persisting only
/// the full-space `x_prev` and reprojecting fixes this: even when
/// `prev.z` rotates between ticks, `v_target` follows.
///
/// Slack rows are anchored at 0 (a healthy WBC has slack ≈ 0 anyway),
/// so the prox cost cleanly biases the equality residual without
/// pinning slack to a stale value.
///
/// Pass `None` on the first tick (no history yet) or when
/// `prox_weight ≤ 0` — the constructor silently falls back to a cold
/// start in either case.
#[derive(Clone, Debug)]
pub struct WarmStart<'a> {
    pub x_prev: Option<&'a DVector<f64>>,
    pub prox_weight: f64,
}

impl<'a> WarmStart<'a> {
    /// Cold start (no warm-start hint, no prox term).
    pub const COLD: WarmStart<'static> = WarmStart {
        x_prev: None,
        prox_weight: 0.0,
    };
}

impl HoQp {
    /// Solve a single-priority problem (no higher tasks).
    pub fn new(task: Task) -> Self {
        Self::new_with_higher_warm(task, None, &WarmStart::COLD)
    }

    /// Solve `task` at the priority level immediately below `higher`.
    /// `higher` may be `None` (this is the top level).
    pub fn new_with_higher(task: Task, higher: Option<&HoQp>) -> Self {
        Self::new_with_higher_warm(task, higher, &WarmStart::COLD)
    }

    /// Like [`HoQp::new_with_higher`] but with a warm-start hint for
    /// the inner QP. See [`WarmStart`].
    pub fn new_with_higher_warm(
        task: Task,
        higher: Option<&HoQp>,
        warm: &WarmStart<'_>,
    ) -> Self {
        let (n_decision_total, prev) = match higher {
            Some(h) => (h.state.x.len(), h.state.clone()),
            None => {
                let n = task.n_decision();
                (
                    n,
                    LevelState {
                        x: DVector::zeros(n),
                        z: DMatrix::identity(n, n),
                        stacked_d: DMatrix::zeros(0, n),
                        stacked_f: DVector::zeros(0),
                        stacked_slack: DVector::zeros(0),
                    },
                )
            }
        };

        debug_assert!(
            task.n_decision() == 0 || task.n_decision() == n_decision_total,
            "HoQp: task decision dim {} does not match accumulated dim {}",
            task.n_decision(),
            n_decision_total,
        );

        let n_v = prev.z.ncols(); // dimension we may still optimise over
        let n_slack = task.n_iq();
        let n_prev_slack = prev.stacked_slack.len();
        let has_eq = task.n_eq() > 0;
        let has_iq = n_slack > 0;

        // ── Build the per-level QP: variables y = [v; w] ─────────────
        let n_y = n_v + n_slack;

        // Hessian: H = block_diag(Z^T A^T A Z + ε·I, I).
        let mut h = DMatrix::zeros(n_y, n_y);
        if has_eq {
            let a_z = &task.a * &prev.z;
            let mut top = a_z.transpose() * &a_z;
            // Tiny ridge on the equality block so a degenerate Hessian
            // doesn't break the inner QP. Matches legged_control's 1e-12.
            //
            // NOTE: tested bumping this to 1e-4 to suppress tick-to-tick
            // QP jitter (min-norm v preference). It works mathematically
            // but penalises ALL variables equally — including τ — so the
            // solver picks (f balancing gravity, τ = 0) and the joint
            // torque collapses. The right fix is per-variable-type
            // weighting (cheap reg on q̈/f, none on τ) or warm-start;
            // both are TODO at the host wiring level.
            for i in 0..n_v {
                top[(i, i)] += 1e-12;
            }
            h.view_mut((0, 0), (n_v, n_v)).copy_from(&top);
        }
        // Slack identity block.
        for i in 0..n_slack {
            h[(n_v + i, n_v + i)] = 1.0;
        }

        // Linear cost: c = [Z^T A^T (A x_prev − b); 0].
        let mut c = DVector::zeros(n_y);
        if has_eq {
            let residual = &task.a * &prev.x - &task.b;
            let top: DVector<f64> = (&task.a * &prev.z).transpose() * residual;
            c.view_mut((0, 0), (n_v, 1)).copy_from(&top);
        }

        // Inequality matrix D̃ y ≤ f̃. Three groups of rows:
        //   1) -w  ≤ 0                         (slack non-negativity)
        //   2) D_{<k} Z_prev v ≤ f_{<k} - D_{<k} x_prev + w_{<k}
        //   3) D_k    Z_prev v - w ≤ f_k - D_k x_prev
        let m_total = n_slack + n_prev_slack + n_slack;
        let mut d_total = DMatrix::zeros(m_total, n_y);
        let mut f_total = DVector::zeros(m_total);
        let mut row = 0usize;

        // Group 1: slack ≥ 0.
        for i in 0..n_slack {
            d_total[(row + i, n_v + i)] = -1.0;
        }
        row += n_slack;

        // Group 2: prior-level inequalities (no new slack — relaxed by
        // already-solved slack stacked into stacked_slack).
        if n_prev_slack > 0 {
            let prev_d_z = &prev.stacked_d * &prev.z;
            d_total
                .view_mut((row, 0), (n_prev_slack, n_v))
                .copy_from(&prev_d_z);
            let rhs = &prev.stacked_f - &prev.stacked_d * &prev.x + &prev.stacked_slack;
            f_total
                .view_mut((row, 0), (n_prev_slack, 1))
                .copy_from(&rhs);
            row += n_prev_slack;
        }

        // Group 3: current-level inequalities with slack.
        if has_iq {
            let d_z = &task.d * &prev.z;
            d_total
                .view_mut((row, 0), (n_slack, n_v))
                .copy_from(&d_z);
            for i in 0..n_slack {
                d_total[(row + i, n_v + i)] = -1.0;
            }
            let rhs = &task.f - &task.d * &prev.x;
            f_total.view_mut((row, 0), (n_slack, 1)).copy_from(&rhs);
        }

        // Solve the inner QP. Use Clarabel via misarta — robust on the
        // moderately-sized (~30 var) problems each level produces, and
        // already a transitive dep through the SRBD MPC.
        //
        // Warm-start: project `warm.x_prev` from the full decision space
        // into this level's basis: `v_target = prev.zᵀ · (x_prev − prev.x)`.
        // misarta's solve_qp then adds a (ρ/2)·‖y − y_target‖² term,
        // anchoring the inner optimum toward `(v_target, 0_slack)`.
        // Re-projecting every tick keeps the anchor valid even though
        // `prev.z` is rebuilt from a `q`-dependent equality matrix.
        let warm_y_owned: Option<DVector<f64>> =
            if warm.prox_weight > 0.0 && n_y > 0 {
                warm.x_prev.and_then(|x_prev| {
                    if x_prev.len() != n_decision_total {
                        return None;
                    }
                    let mut y = DVector::zeros(n_y);
                    if n_v > 0 {
                        let delta = x_prev - &prev.x;
                        let v_target = prev.z.transpose() * delta;
                        for i in 0..n_v {
                            y[i] = v_target[i];
                        }
                    }
                    // Slack rows stay at 0 — a converged WBC has slack ≈ 0
                    // and we don't want to anchor toward a transient
                    // violation from the previous tick.
                    Some(y)
                })
            } else {
                None
            };
        let cfg = QpConfig {
            solver: QpSolver::Clarabel,
            prox_weight: if warm_y_owned.is_some() {
                warm.prox_weight
            } else {
                0.0
            },
            ..Default::default()
        };
        let qp_a_iq = (m_total > 0).then(|| d_total.clone());
        let qp_b_iq = (m_total > 0).then(|| f_total.clone());
        let sol = solve_qp(
            &h,
            &c,
            None,
            None,
            qp_a_iq.as_ref(),
            qp_b_iq.as_ref(),
            warm_y_owned.as_ref(),
            &cfg,
        );
        if !matches!(sol.status, QpStatus::Optimal) {
            log::warn!(
                "HoQp inner QP did not reach optimal: {:?} (iter={})",
                sol.status,
                sol.iterations
            );
        }

        // Extract `v` and `w` from the solver result. If the QP failed
        // we still return `state.x = prev.x` (no progress) so the
        // overall WBC degrades gracefully rather than panicking.
        let (v_step, slack_curr, y_full) = if matches!(sol.status, QpStatus::Optimal) {
            let v = sol.x.rows(0, n_v).into_owned();
            let w = sol.x.rows(n_v, n_slack).into_owned();
            (v, w, sol.x.clone())
        } else {
            (
                DVector::zeros(n_v),
                DVector::zeros(n_slack),
                DVector::zeros(n_y),
            )
        };

        // Update cumulative solution: x_k = x_{k-1} + Z_{k-1} · v.
        let x_new = &prev.x + &prev.z * &v_step;

        // Refine the null space for the next level: Z_k = Z_{k-1} · null(A_k Z_{k-1}).
        let z_new = if has_eq {
            let m = &task.a * &prev.z;
            let n = right_null_space(&m);
            if n.ncols() == 0 {
                // No remaining freedom — subsequent tasks can only emit
                // slack. Keep an empty (n_total × 0) matrix so the next
                // level's `n_v` collapses to zero.
                DMatrix::zeros(n_decision_total, 0)
            } else {
                &prev.z * n
            }
        } else {
            prev.z.clone()
        };

        // Stacked inequalities for the next level.
        let stacked_d = stack_rows(&prev.stacked_d, &task.d, n_decision_total);
        let stacked_f = stack_vec(&prev.stacked_f, &task.f);
        let stacked_slack = stack_vec(&prev.stacked_slack, &slack_curr);

        Self {
            state: LevelState {
                x: x_new,
                z: z_new,
                stacked_d,
                stacked_f,
                stacked_slack,
            },
            y_solution: y_full,
        }
    }

    /// Final solution `x` after all levels.
    pub fn solution(&self) -> &DVector<f64> {
        &self.state.x
    }

    /// Null-space basis after all levels (for diagnostic / chaining).
    pub fn null_space(&self) -> &DMatrix<f64> {
        &self.state.z
    }

    /// Inner-QP decision vector `y = [v; w]` solved at this level.
    /// Caller passes this back to the **same** level on the next tick
    /// (via [`WarmStart::y_prev`]) to anchor the optimum and dampen
    /// jitter.
    pub fn y_solution(&self) -> &DVector<f64> {
        &self.y_solution
    }
}

/// Compute a basis for the right null space of `m` (`m·z = 0`).
/// Returns an `(n_cols × k)` matrix whose columns span the kernel,
/// where `k = n_cols − rank(m)`.
///
/// We diagonalise the symmetric Gram matrix `Gᵀ G` (cols × cols) and
/// pick eigenvectors with near-zero eigenvalues as the kernel basis.
/// This gives the **full** column-space of V regardless of rows-vs-cols
/// shape — nalgebra's thin SVD truncates V when `rows < cols`, which
/// is exactly the case the WBC hits (few task rows, many decision
/// variables).
fn right_null_space(m: &DMatrix<f64>) -> DMatrix<f64> {
    let (rows, cols) = m.shape();
    if rows == 0 {
        return DMatrix::identity(cols, cols);
    }
    if cols == 0 {
        // No columns → no null space directions to span. Returns a
        // (0 × 0) matrix so the caller's Z update collapses cleanly to
        // an empty matrix (the next priority level inherits zero
        // freedom, which is correct).
        return DMatrix::zeros(0, 0);
    }
    let g = m.transpose() * m;
    let eig = g.symmetric_eigen();
    // Eigenvalues are singular-values-squared. Threshold **relative to
    // the spectrum**: forming the Gram matrix `mᵀm` squares the
    // condition number, so an exact-zero singular value surfaces as an
    // eigenvalue near `λ_max · ε` (~1e-16·λ_max), not near 0. An
    // absolute `(ε·σ_max)²` tol therefore fails to catch it on
    // ill-conditioned data and *undercounts* the null space — starving
    // lower-priority tasks of freedom. A relative `1e-9·λ_max` cleanly
    // separates the numerically-zero eigenvalues from the genuine ones
    // (structured robot Jacobians rank identically under either rule —
    // the gap is far wider). An absolute floor keeps the all-zero case
    // sane.
    let lambda_max = eig
        .eigenvalues
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(0.0);
    let abs_floor = {
        let s = (rows.max(cols) as f64) * f64::EPSILON;
        s * s
    };
    let tol2 = (1e-9 * lambda_max).max(abs_floor);
    // Collect indices of eigenvectors with eigenvalue ≈ 0.
    let kernel_cols: Vec<usize> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .filter(|&(_, &lam)| lam <= tol2)
        .map(|(i, _)| i)
        .collect();
    let k = kernel_cols.len();
    if k == 0 {
        return DMatrix::zeros(cols, 0);
    }
    let mut z = DMatrix::zeros(cols, k);
    for (j, &i) in kernel_cols.iter().enumerate() {
        let v = eig.eigenvectors.column(i);
        for r in 0..cols {
            z[(r, j)] = v[r];
        }
    }
    z
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
    use approx::assert_relative_eq;

    /// Single-priority unconstrained least-squares: HoQp should reduce
    /// to the standard pseudoinverse solution.
    #[test]
    fn single_task_equality_solves_least_squares() {
        // min ‖A x − b‖² with A = [[1, 0], [0, 1]], b = [3, 4] → x = [3, 4]
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_vec(vec![3.0, 4.0]);
        let task = Task::equality(a, b);
        let sol = HoQp::new(task);
        assert_relative_eq!(sol.solution()[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(sol.solution()[1], 4.0, epsilon = 1e-6);
    }

    /// Two-level hierarchy: high-priority equality `x[0] = 1` is
    /// strict; low-priority equality `x[1] = 2 · x[0]` then has full
    /// freedom in the remaining null space and must hit `x[1] = 2`.
    #[test]
    fn hierarchical_two_levels_respect_priority() {
        // Level 0: x[0] = 1
        let task_0 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DVector::from_vec(vec![1.0]),
        );
        // Level 1: 2·x[0] − x[1] = 0 ⇒ x[1] = 2·x[0] = 2
        let task_1 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[2.0, -1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let l0 = HoQp::new(task_0);
        let l1 = HoQp::new_with_higher(task_1, Some(&l0));
        assert_relative_eq!(l1.solution()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(l1.solution()[1], 2.0, epsilon = 1e-6);
    }

    /// Hard inequality at top priority must be respected even when a
    /// lower-priority equality wants to violate it.
    #[test]
    fn higher_inequality_blocks_lower_equality() {
        // Level 0: x[0] ≤ 1 (hard)
        let task_0 = Task::inequality(
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DVector::from_vec(vec![1.0]),
        );
        // Level 1: x[0] = 5 (would violate)
        let task_1 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DVector::from_vec(vec![5.0]),
        );
        let l0 = HoQp::new(task_0);
        let l1 = HoQp::new_with_higher(task_1, Some(&l0));
        // Slack must absorb the violation; x[0] should be at the bound.
        // 1e-4 tolerance accounts for clarabel's interior-point convergence
        // criteria — the hard inequality is respected to that precision,
        // far below any meaningful constraint violation in joint torque
        // / friction-cone units.
        assert!(
            l1.solution()[0] <= 1.0 + 1e-4,
            "x[0] = {} should be ≤ 1 (hard ineq)",
            l1.solution()[0],
        );
    }

    /// Warm-start: a single-priority QP with a wide null space (rank-
    /// deficient task) should prefer solutions close to `x_prev` when
    /// a positive `prox_weight` is supplied.
    #[test]
    fn warm_start_anchors_solution_in_null_space() {
        // Task with a rank-1 equality on 3 variables: `x[0] + x[1] = 2`.
        // Solutions form the 2-D plane (1-α, 1+α, anything). Without
        // warm-start the lowest-norm solution has x[2] = 0; with warm-
        // start anchored at (1, 1, 5) the optimum should snap to that.
        let a = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 0.0]);
        let b = DVector::from_vec(vec![2.0]);
        let task = || Task::equality(a.clone(), b.clone());

        // Cold: minimum-norm solution sits near (1, 1, 0).
        let cold = HoQp::new(task());
        assert_relative_eq!(cold.solution()[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(cold.solution()[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(cold.solution()[2], 0.0, epsilon = 1e-4);

        // Warm-anchored at (1, 1, 5): equality satisfied, x[2] free →
        // pulled to 5.
        let x_prev = DVector::from_vec(vec![1.0, 1.0, 5.0]);
        let warm = WarmStart {
            x_prev: Some(&x_prev),
            prox_weight: 1.0, // strong: we want a measurable shift
        };
        let warm_sol = HoQp::new_with_higher_warm(task(), None, &warm);
        // Equality is hard at this priority — must still be satisfied.
        assert_relative_eq!(
            warm_sol.solution()[0] + warm_sol.solution()[1],
            2.0,
            epsilon = 1e-4,
        );
        // x[2] (orthogonal to the equality row) follows the anchor.
        assert_relative_eq!(warm_sol.solution()[2], 5.0, epsilon = 1e-2);
    }

    /// Warm-start with `prox_weight = 0` must produce identical results
    /// to a cold start (backward compat).
    #[test]
    fn warm_start_zero_prox_is_a_noop() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_vec(vec![3.0, 4.0]);
        let cold = HoQp::new(Task::equality(a.clone(), b.clone()));
        let x_prev = DVector::from_vec(vec![100.0, 100.0]); // far anchor
        let warm = WarmStart {
            x_prev: Some(&x_prev),
            prox_weight: 0.0,
        };
        let warm_sol =
            HoQp::new_with_higher_warm(Task::equality(a, b), None, &warm);
        for i in 0..2 {
            assert_relative_eq!(
                cold.solution()[i],
                warm_sol.solution()[i],
                epsilon = 1e-6,
            );
        }
    }

    /// Lower-priority task that's compatible with the higher one is
    /// solved to optimum without disturbing the upper.
    #[test]
    fn compatible_lower_task_is_satisfied_exactly() {
        // Level 0: x[0] = 2
        let task_0 = Task::equality(
            DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]),
            DVector::from_vec(vec![2.0]),
        );
        // Level 1: x[1] = 5 (orthogonal to level 0)
        let task_1 = Task::equality(
            DMatrix::from_row_slice(1, 3, &[0.0, 1.0, 0.0]),
            DVector::from_vec(vec![5.0]),
        );
        let l0 = HoQp::new(task_0);
        let l1 = HoQp::new_with_higher(task_1, Some(&l0));
        assert_relative_eq!(l1.solution()[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(l1.solution()[1], 5.0, epsilon = 1e-6);
    }
}
