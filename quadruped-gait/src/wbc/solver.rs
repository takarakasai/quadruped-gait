//! Formulation-switchable WBC solve (the misa-wbc `Dynamics` path).
//!
//! [`WbcSolver`] builds the exact task stack of
//! [`solve_warm_with_weights`](super::wbc::solve_warm_with_weights) —
//! same tasks, same priorities, same weights — but through misa-wbc's
//! [`Dynamics`] context, so the decision-variable **formulation**
//! (explicit / OpenSoT accel-space / GID force-space), the hierarchy
//! **strategy** and the QP **backend** are all runtime switches, and a
//! persistent [`misa_wbc::Solver`] session warm-starts the active-set
//! backend across ticks.
//!
//! The legacy path stays the default everywhere; hosts opt in by
//! holding a `WbcSolver` and calling [`WbcSolver::solve`] instead.

use misa_wbc::{tasks, Dynamics, SolveConfig, Task};
use nalgebra::{DMatrix, DVector};

pub use misa_wbc::{Formulation, HqpStrategy, QpSolver};

use super::wbc::{WbcInputs, WbcSolution, WbcWarmStart, WbcWeights};

/// A persistent, mode-switchable WBC solver over the misa-wbc task
/// catalogue. Construct once, call [`WbcSolver::solve`] every tick.
#[derive(Debug)]
pub struct WbcSolver {
    formulation: Formulation,
    cfg: SolveConfig,
    session: misa_wbc::Solver,
}

impl WbcSolver {
    pub fn new(formulation: Formulation, cfg: SolveConfig) -> Self {
        WbcSolver { formulation, cfg, session: misa_wbc::Solver::new() }
    }

    pub fn formulation(&self) -> Formulation {
        self.formulation
    }

    /// Drop all warm-start state (e.g. after a teleport / reset).
    pub fn reset(&mut self) {
        self.session.reset();
    }

    /// One WBC tick. Mirrors `solve_warm_with_weights` task-for-task;
    /// the warm anchor is the session's own previous solution (in this
    /// formulation's layout), weighted by `warm.prox_weight` —
    /// `warm.x_prev` is not needed and is ignored.
    pub fn solve(
        &mut self,
        inputs: &WbcInputs<'_>,
        warm: &WbcWarmStart<'_>,
        w: &WbcWeights,
    ) -> WbcSolution {
        let dims = inputs.dims;
        let (nv, nc, na) = (dims.nv, dims.nc, dims.na);
        let dyn_ = Dynamics::new(self.formulation, inputs.mass, inputs.nle, inputs.j_contact, na);
        let n = dyn_.layout().n_decision();
        let f = dyn_.forces();
        let q = dyn_.qddot();
        let tau = dyn_.tau();

        // ── Priority 0: hard constraints ───────────────────────────
        let mut p0 = Task::empty(n);
        if let Some(phys) = dyn_.dynamics_task() {
            p0 = p0 + phys.weight(w.floating_base_eom);
        }
        p0 = p0 + tasks::box_bound(tau, inputs.torque_max);
        // Per-contact: stance → friction pyramid; swing → zero force.
        for c in 0..nc {
            let mut sel = DMatrix::zeros(3, 3 * nc);
            for i in 0..3 {
                sel[(i, 3 * c + i)] = 1.0;
            }
            let fc = &sel * &f;
            p0 = p0
                + if inputs.contact_flag[c] {
                    tasks::friction_pyramid(&fc, inputs.friction_mu)
                } else {
                    tasks::track(&fc, &DVector::zeros(3))
                };
        }
        // Stance feet hold still: J_st·q̈ + (J̇v)_st = 0.
        let n_st: usize = inputs.contact_flag.iter().filter(|&&b| b).count();
        if n_st > 0 {
            let mut jc_st = DMatrix::zeros(3 * n_st, nv);
            let mut djv_st = DVector::zeros(3 * n_st);
            let mut r = 0;
            for c in 0..nc {
                if !inputs.contact_flag[c] {
                    continue;
                }
                jc_st.rows_mut(r, 3).copy_from(&inputs.j_contact.rows(3 * c, 3));
                djv_st.rows_mut(r, 3).copy_from(&inputs.dj_v.rows(3 * c, 3));
                r += 3;
            }
            p0 = p0
                + tasks::zero_contact_acceleration(q, &jc_st, &djv_st)
                    .weight(w.no_contact_motion);
        }

        // ── Priority 1: motion tracking ────────────────────────────
        let mut sel_base = DMatrix::zeros(6, nv);
        for i in 0..6 {
            sel_base[(i, i)] = 1.0;
        }
        let mut p1 =
            tasks::track(&(&sel_base * q), inputs.a_base_des).weight(w.base_accel);
        let n_sw: usize = inputs.swing_actuator_flag.iter().filter(|&&b| b).count();
        if n_sw > 0 {
            let mut sel = DMatrix::zeros(n_sw, nv);
            let mut des = DVector::zeros(n_sw);
            let mut r = 0;
            for i in 0..na {
                if !inputs.swing_actuator_flag[i] {
                    continue;
                }
                sel[(r, 6 + i)] = 1.0;
                des[r] = inputs.swing_q_ddot_des[i];
                r += 1;
            }
            p1 = p1 + tasks::track(&(&sel * q), &des).weight(w.swing_leg);
        }

        // ── Priority 2: GRF + τ_grav regularisation ────────────────
        let p2 = tasks::regularize(&f, inputs.f_grf_des).weight(w.contact_force)
            + tasks::track(tau, inputs.tau_gravity).weight(w.tau_gravity);

        let mut cfg = self.cfg.clone();
        cfg.prox_weight = warm.prox_weight;
        let sol = self
            .session
            .solve(&[p0, p1, p2], &cfg)
            .expect("WbcSolver: level dimensions are consistent by construction");

        // Extract the physical triple and re-assemble the explicit
        // x_full = [q̈; f; τ] the host caches.
        let e = dyn_.extract(&sol.x);
        let mut x_full = DVector::zeros(nv + 3 * nc + na);
        x_full.rows_mut(0, nv).copy_from(&e.qddot);
        x_full.rows_mut(nv, 3 * nc).copy_from(&e.forces);
        x_full.rows_mut(nv + 3 * nc, na).copy_from(&e.tau);
        WbcSolution { q_ddot: e.qddot, f_grf: e.forces, tau: e.tau, x_full }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wbc::{wbc, WbcDims};

    /// A consistent Go2-sized fixture with mixed stance/swing.
    struct Fix {
        mass: DMatrix<f64>,
        nle: DVector<f64>,
        jc: DMatrix<f64>,
        dj_v: DVector<f64>,
        torque_max: DVector<f64>,
        a_base_des: DVector<f64>,
        swing_des: DVector<f64>,
        swing_flag: Vec<bool>,
        f_des: DVector<f64>,
        tau_g: DVector<f64>,
    }

    fn fix() -> (WbcDims, Fix) {
        let dims = WbcDims { nv: 18, nc: 4, na: 12 };
        let (nv, nc, na) = (dims.nv, dims.nc, dims.na);
        let l = DMatrix::from_fn(nv, 3, |i, j| 0.1 * ((i + 2 * j) as f64 * 0.37).sin());
        let mass = DMatrix::<f64>::identity(nv, nv) * 1.2 + &l * l.transpose();
        let mut nle = DVector::zeros(nv);
        nle[2] = 15.0 * 9.81;
        let jc = DMatrix::from_fn(3 * nc, nv, |i, j| {
            if j < 3 && i % 3 == j { 1.0 } else { 0.15 * ((i * 5 + j * 3) as f64 * 0.23).cos() }
        });
        let dj_v = DVector::from_fn(3 * nc, |i, _| 0.03 * (i as f64 * 1.1).sin());
        let mut swing_flag = vec![false; na];
        for i in 9..12 {
            swing_flag[i] = true; // foot 3 swings
        }
        let f = Fix {
            mass,
            nle,
            jc,
            dj_v,
            torque_max: DVector::from_element(na, 23.7),
            a_base_des: DVector::from_fn(6, |i, _| 0.1 * (i as f64 - 2.0)),
            swing_des: DVector::from_fn(na, |i, _| if i >= 9 { 0.5 } else { 0.0 }),
            swing_flag,
            f_des: DVector::from_fn(3 * nc, |i, _| if i % 3 == 2 { 36.8 } else { 0.0 }),
            tau_g: DVector::from_fn(na, |i, _| 0.2 * (i as f64 * 0.7).cos()),
        };
        (dims, f)
    }

    fn inputs<'a>(dims: WbcDims, f: &'a Fix) -> WbcInputs<'a> {
        WbcInputs {
            dims,
            mass: &f.mass,
            nle: &f.nle,
            j_contact: &f.jc,
            dj_v: &f.dj_v,
            contact_flag: [true, true, true, false],
            friction_mu: 0.6,
            torque_max: &f.torque_max,
            a_base_des: &f.a_base_des,
            swing_q_ddot_des: &f.swing_des,
            swing_actuator_flag: &f.swing_flag,
            f_grf_des: &f.f_des,
            tau_gravity: &f.tau_g,
        }
    }

    /// The Dynamics path in the explicit formulation reproduces the
    /// legacy solve on the same inputs and weights.
    #[test]
    fn explicit_matches_legacy_solve() {
        let (dims, fx) = fix();
        let inp = inputs(dims, &fx);
        let warm = WbcWarmStart::default();
        let w = WbcWeights::default();

        let legacy = wbc::solve_warm_with_weights(&inp, &warm, &w);
        let mut solver = WbcSolver::new(Formulation::Explicit, SolveConfig::default());
        let new = solver.solve(&inp, &warm, &w);

        assert!(
            (&new.q_ddot - &legacy.q_ddot).norm() < 1e-4,
            "q̈ differs: {}",
            (&new.q_ddot - &legacy.q_ddot).norm()
        );
        assert!((&new.f_grf - &legacy.f_grf).norm() < 1e-4, "f differs");
        assert!((&new.tau - &legacy.tau).norm() < 1e-4, "τ differs");
    }

    /// All three formulations land on the same physical solution.
    #[test]
    fn formulations_agree_on_wbc_inputs() {
        let (dims, fx) = fix();
        let inp = inputs(dims, &fx);
        let warm = WbcWarmStart::default();
        let w = WbcWeights::default();

        let mut reference: Option<WbcSolution> = None;
        for form in [Formulation::Explicit, Formulation::AccelSpace, Formulation::ForceSpace] {
            let mut solver = WbcSolver::new(form, SolveConfig::default());
            let sol = solver.solve(&inp, &warm, &w);
            if let Some(r) = &reference {
                assert!(
                    (&sol.q_ddot - &r.q_ddot).norm() < 1e-3,
                    "{form:?}: q̈ differs by {}",
                    (&sol.q_ddot - &r.q_ddot).norm()
                );
                assert!((&sol.tau - &r.tau).norm() < 1e-3, "{form:?}: τ differs");
            } else {
                reference = Some(sol);
            }
        }
    }

    /// The GID-mode combination (force space + active set) solves and
    /// satisfies the physics on WBC inputs.
    #[test]
    fn force_space_active_set_is_physical() {
        let (dims, fx) = fix();
        let inp = inputs(dims, &fx);
        let cfg = SolveConfig { backend: QpSolver::ActiveSet, ..Default::default() };
        let mut solver = WbcSolver::new(Formulation::ForceSpace, cfg);
        let sol = solver.solve(&inp, &WbcWarmStart::default(), &WbcWeights::default());

        // EoM holds structurally.
        let mut s_t = DMatrix::zeros(dims.nv, dims.na);
        for i in 0..dims.na {
            s_t[(dims.nv - dims.na + i, i)] = 1.0;
        }
        let eom = &fx.mass * &sol.q_ddot + &fx.nle
            - &s_t * &sol.tau
            - fx.jc.transpose() * &sol.f_grf;
        assert!(eom.norm() < 1e-6, "EoM violated: {}", eom.norm());
        // Torques within limits.
        assert!(sol.tau.amax() <= 23.7 + 1e-6);
    }
}
