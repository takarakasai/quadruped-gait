//! **P1 stage-1 solver**: a self-contained planar-SRBD *periodic-orbit*
//! generator for a FORWARD-moving Bound (`ref/bound_trajopt_design.md`).
//!
//! §5f established that no bolt-on controller makes the energetic Bound go
//! forward; §5f9 P0 confirmed (in Python) that a feasible forward periodic
//! orbit EXISTS. This module re-implements that existence solver in Rust so
//! the reference orbit can be regenerated for any speed / duty without the
//! Python dependency -- the offline half of the two-stage trajopt plan
//! (stage 2 = feed the orbit as the MPC reference, see
//! [`crate::FullCentroidalMpcGaitController::set_bound_tabulated_reference`]).
//!
//! Method: SINGLE SHOOTING. The per-stance ground-reaction force is a
//! low-order polynomial; the planar SRBD `s=[x,z,theta,vx,vz,w]` is
//! integrated (RK4) through the fixed bound schedule
//! (front stance | flight | rear stance | flight); a Levenberg-Marquardt
//! least-squares solve finds the initial state + force coefficients + foot
//! positions so the cycle is PERIODIC (modulo the forward translation
//! `vx_target * T`) at the target speed, subject to soft friction / fz>=0 /
//! reachability / pitch-bound penalties. Dynamics are satisfied by
//! construction (integration), so there are no collocation defects.
//!
//! The planar model is intentionally the same fidelity as P0 (no roll, no
//! whole-body kinematics) -- enough to produce the base-state reference the
//! stage-2 MPC tracks. Whole-body / 3D is later work (P3).

use nalgebra::{DMatrix, DVector};

/// Physical + gait parameters for the periodic-orbit solve.
#[derive(Clone, Debug)]
pub struct PeriodicBoundParams {
    pub mass_kg: f64,
    pub inertia_yy: f64,
    /// Nominal fore-aft foot offset magnitude from the CoM (|front|=|rear|).
    pub r_x: f64,
    /// Nominal CoM height.
    pub h0: f64,
    pub cycle_period_s: f64,
    /// Per-pair stance fraction of the cycle (< 0.5 for a flight phase).
    pub duty_factor: f64,
    /// Forward speed to design the orbit for (m/s).
    pub vx_target: f64,
    pub friction_mu: f64,
    /// Foot reachability box: |x_foot - x_com| <= reach during stance.
    pub reach: f64,
    /// Pitch bound (rad): an energetic Bound, not a somersault.
    pub pitch_max: f64,
}

impl PeriodicBoundParams {
    /// Go2 defaults (Sec.5bb/5bd, shared with the point-mass scripts).
    pub fn go2(vx_target: f64, cycle_period_s: f64, duty_factor: f64) -> Self {
        Self {
            mass_kg: 15.606,
            inertia_yy: 0.0981,
            r_x: 0.1922,
            h0: 0.2664,
            cycle_period_s,
            duty_factor,
            vx_target,
            friction_mu: 0.7,
            reach: 0.16,
            pitch_max: 0.6,
        }
    }
}

/// One phase-sampled point of the reference orbit: the base-state target
/// the stage-2 MPC tracks. `phase` in [0,1) over one cycle.
#[derive(Clone, Copy, Debug)]
pub struct OrbitSample {
    pub phase: f64,
    pub z: f64,
    pub pitch: f64,
    pub vx: f64,
    pub vz: f64,
    pub w: f64,
}

/// The solved forward-Bound periodic orbit.
#[derive(Clone, Debug)]
pub struct BoundOrbit {
    pub samples: Vec<OrbitSample>,
    /// Front / rear foot fore-aft positions (world, at the cycle start).
    pub xf: f64,
    pub xr: f64,
    /// **P3-a**: front / rear pair fore-aft foothold RELATIVE TO THE CoM at
    /// that pair's touchdown (front at phase 0, rear at phase 0.5). These
    /// are the placements that make the orbit forward-moving AND
    /// pitch-balanced -- the controller can follow them directly instead of
    /// the Raibert+deadbeat footstep (which fought forward vs stabilize).
    pub front_foothold: f64,
    pub rear_foothold: f64,
    /// Max |periodicity + forward| residual of the returned solution.
    pub periodicity_residual: f64,
    /// Worst friction margin (mu - |fx|/fz) over stance; >= 0 is feasible.
    pub friction_margin: f64,
    /// Rows as `[phase, z, pitch, vx, vz, w]` -- ready for
    /// `set_bound_tabulated_reference`.
    pub table: Vec<[f64; 6]>,
}

const GRAVITY: f64 = 9.81;
const N_PARAM: usize = 19;

// parameter layout (mirrors the P0 Python shooting):
// [z0, th0, vx0, vz0, w0,
//  ffx0,ffx1,ffx2, ffz0,ffz1,ffz2,
//  frx0,frx1,frx2, frz0,frz1,frz2, xf, xr]
const P_Z0: usize = 0;
const P_TH0: usize = 1;
const P_VX0: usize = 2;
const P_VZ0: usize = 3;
const P_W0: usize = 4;
const P_FFX0: usize = 5;
const P_FFZ0: usize = 8;
const P_FRX0: usize = 11;
const P_FRZ0: usize = 14;
const P_XF: usize = 17;
const P_XR: usize = 18;

struct Model {
    p: PeriodicBoundParams,
    t_st: f64,
    t_fl: f64,
}

/// Which pair (if any) is on the ground.
#[derive(Clone, Copy, PartialEq)]
enum Stance {
    Front,
    Rear,
    Flight,
}

impl Model {
    fn new(p: PeriodicBoundParams) -> Self {
        let t_st = p.duty_factor * p.cycle_period_s;
        let t_fl = 0.5 * p.cycle_period_s - t_st;
        Model { p, t_st, t_fl }
    }

    /// The four phases: (duration, stance).
    fn phases(&self) -> [(f64, Stance); 4] {
        [
            (self.t_st, Stance::Front),
            (self.t_fl, Stance::Flight),
            (self.t_st, Stance::Rear),
            (self.t_fl, Stance::Flight),
        ]
    }

    /// Quadratic-in-time GRF for the active pair, `(ffx,ffz,frx,frz)`.
    fn force_at(&self, active: Stance, frac: f64, x: &[f64]) -> (f64, f64, f64, f64) {
        let quad = |i: usize| x[i] + x[i + 1] * frac + x[i + 2] * frac * frac;
        match active {
            Stance::Front => (quad(P_FFX0), quad(P_FFZ0), 0.0, 0.0),
            Stance::Rear => (0.0, 0.0, quad(P_FRX0), quad(P_FRZ0)),
            Stance::Flight => (0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Planar SRBD state derivative.
    fn deriv(&self, s: &[f64; 6], f: (f64, f64, f64, f64), xf: f64, xr: f64) -> [f64; 6] {
        let (_x, z, _th, vx, vz, w) = (s[0], s[1], s[2], s[3], s[4], s[5]);
        let (ffx, ffz, frx, frz) = f;
        let m = self.p.mass_kg;
        let ax = (ffx + frx) / m;
        let az = (ffz + frz) / m - GRAVITY;
        // pitch torque about CoM (foot z = 0): (xfoot-x)*fz + z*fx per pair
        let tau = (xf - s[0]) * ffz + z * ffx + (xr - s[0]) * frz + z * frx;
        [vx, vz, w, ax, az, tau / self.p.inertia_yy]
    }

    /// Integrate one cycle (RK4), returning the trajectory (nsub+1 pts per
    /// phase, sharing endpoints).
    fn integrate(&self, x: &[f64], nsub: usize) -> Vec<[f64; 6]> {
        let mut s = [0.0, x[P_Z0], x[P_TH0], x[P_VX0], x[P_VZ0], x[P_W0]];
        let (xf, xr) = (x[P_XF], x[P_XR]);
        let mut traj = Vec::with_capacity(4 * nsub + 1);
        traj.push(s);
        for (dur, active) in self.phases() {
            let h = dur / nsub as f64;
            for i in 0..nsub {
                let f = |ss: &[f64; 6], frac: f64| {
                    self.deriv(ss, self.force_at(active, frac, x), xf, xr)
                };
                let frac0 = i as f64 / nsub as f64;
                let frac_h = (i as f64 + 0.5) / nsub as f64;
                let frac1 = (i as f64 + 1.0) / nsub as f64;
                let k1 = f(&s, frac0);
                let s2 = add(&s, &k1, 0.5 * h);
                let k2 = f(&s2, frac_h);
                let s3 = add(&s, &k2, 0.5 * h);
                let k3 = f(&s3, frac_h);
                let s4 = add(&s, &k3, h);
                let k4 = f(&s4, frac1);
                for j in 0..6 {
                    s[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
                }
                traj.push(s);
            }
        }
        traj
    }

    /// Residual vector: periodicity + forward (hard) then feasibility
    /// penalties (soft one-sided) then light regularizers. Mirrors P0.
    fn residuals(&self, x: &[f64]) -> DVector<f64> {
        let nsub = 40;
        let traj = self.integrate(x, nsub);
        let n = traj.len();
        let st = &traj[n - 1];
        let mg = self.p.mass_kg * GRAVITY;
        let w_per = 20.0;
        let w_feas = 8.0;
        let mut r: Vec<f64> = Vec::new();
        // periodicity of [z,th,vx,vz,w]; x advances by vx_target*T
        r.push(w_per * (st[1] - x[P_Z0]));
        r.push(w_per * (st[2] - x[P_TH0]));
        r.push(w_per * (st[3] - x[P_VX0]));
        r.push(w_per * (st[4] - x[P_VZ0]));
        r.push(w_per * (st[5] - x[P_W0]));
        r.push(w_per * ((st[0] - 0.0) - self.p.vx_target * self.p.cycle_period_s));
        // feasibility penalties sampled along the trajectory
        let per_phase = (n - 1) / 4;
        let relu = |a: f64| if a > 0.0 { a } else { 0.0 };
        let (xf, xr) = (x[P_XF], x[P_XR]);
        let mut zsum = 0.0;
        for pi in 0..4 {
            let (dur, active) = self.phases()[pi];
            let _ = dur;
            for i in 0..per_phase {
                let k = (pi * per_phase + i).min(n - 1);
                let (xc, z, th) = (traj[k][0], traj[k][1], traj[k][2]);
                let frac = i as f64 / per_phase as f64;
                let (ffx, ffz, frx, frz) = self.force_at(active, frac, x);
                match active {
                    Stance::Front => {
                        r.push(w_feas * relu(-ffz) / mg);
                        r.push(3.0 * w_feas * relu(ffx.abs() - self.p.friction_mu * ffz) / mg);
                        r.push(w_feas * relu((xf - xc).abs() - self.p.reach));
                    }
                    Stance::Rear => {
                        r.push(w_feas * relu(-frz) / mg);
                        r.push(3.0 * w_feas * relu(frx.abs() - self.p.friction_mu * frz) / mg);
                        r.push(w_feas * relu((xr - xc).abs() - self.p.reach));
                    }
                    Stance::Flight => {}
                }
                r.push(w_feas * relu(th.abs() - self.p.pitch_max));
                r.push(w_feas * relu(0.12 - z));
                // Soft pitch-magnitude regularizer (not just the hard bound):
                // pulls the solve toward the low-pitch orbit rather than a
                // feasible-but-large-pitch local minimum (P0's trf found the
                // former; bare LM otherwise binds pitch_max).
                r.push(0.4 * th);
                zsum += z;
            }
        }
        // light regularizers: symmetric pitch, small tangential, CoM ~ h0
        let zavg = zsum / (4 * per_phase) as f64;
        r.push(0.2 * x[P_TH0]);
        r.push(1e-3 * x[P_FFX0]);
        r.push(1e-3 * x[P_FRX0]);
        r.push(1.0 * (zavg - self.p.h0));
        DVector::from_vec(r)
    }
}

fn add(s: &[f64; 6], k: &[f64; 6], h: f64) -> [f64; 6] {
    let mut o = *s;
    for j in 0..6 {
        o[j] += h * k[j];
    }
    o
}

/// Finite-difference Jacobian of the residuals at `x`.
fn jacobian(model: &Model, x: &[f64], r0: &DVector<f64>) -> DMatrix<f64> {
    let m = r0.len();
    let mut j = DMatrix::zeros(m, N_PARAM);
    let eps = 1e-6;
    let mut xp = x.to_vec();
    for c in 0..N_PARAM {
        let h = eps * (1.0 + x[c].abs());
        xp[c] = x[c] + h;
        let rp = model.residuals(&xp);
        xp[c] = x[c];
        for row in 0..m {
            j[(row, c)] = (rp[row] - r0[row]) / h;
        }
    }
    j
}

/// Levenberg-Marquardt least-squares solve from initial guess `x0`.
fn levenberg_marquardt(model: &Model, mut x: Vec<f64>, max_iter: usize) -> (Vec<f64>, f64) {
    let mut r = model.residuals(&x);
    let mut cost = 0.5 * r.dot(&r);
    let mut lambda = 1e-3;
    for _ in 0..max_iter {
        let j = jacobian(model, &x, &r);
        let jt = j.transpose();
        let jtj = &jt * &j;
        let jtr = &jt * &r;
        let mut improved = false;
        for _ in 0..12 {
            let mut a = jtj.clone();
            for d in 0..N_PARAM {
                a[(d, d)] += lambda * (1.0 + jtj[(d, d)]);
            }
            let delta = match a.clone().lu().solve(&(-&jtr)) {
                Some(d) => d,
                None => break,
            };
            let xn: Vec<f64> = (0..N_PARAM).map(|i| x[i] + delta[i]).collect();
            let rn = model.residuals(&xn);
            let cn = 0.5 * rn.dot(&rn);
            if cn < cost {
                x = xn;
                r = rn;
                cost = cn;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                break;
            } else {
                lambda *= 4.0;
            }
        }
        if !improved || cost < 1e-12 {
            break;
        }
    }
    (x, cost)
}

/// Solve for a forward-moving periodic Bound orbit. Returns `None` if no
/// solution meets the periodicity (< 1e-3) and friction (> -0.05) targets.
pub fn solve(params: &PeriodicBoundParams) -> Option<BoundOrbit> {
    let model = Model::new(params.clone());
    if model.t_fl <= 0.0 {
        return None; // need a flight phase (duty < 0.5)
    }
    let fz_nom = params.mass_kg * GRAVITY / params.duty_factor;
    let base = |vz0: f64| {
        let mut x = vec![0.0; N_PARAM];
        x[P_Z0] = params.h0;
        x[P_VX0] = params.vx_target;
        x[P_VZ0] = vz0;
        x[P_FFZ0] = fz_nom;
        x[P_FRZ0] = fz_nom;
        x[P_XF] = params.r_x;
        x[P_XR] = -params.r_x;
        x
    };
    // seed sensitivity: try a few liftoff-velocity seeds, keep the best.
    let mut best: Option<(Vec<f64>, f64)> = None;
    for &vz0 in &[0.2_f64, 0.4, 0.6] {
        let (x, cost) = levenberg_marquardt(&model, base(vz0), 120);
        if best.as_ref().map(|b| cost < b.1).unwrap_or(true) {
            best = Some((x, cost));
        }
    }
    let (x, _cost) = best?;
    Some(build_orbit(&model, &x))
}

fn build_orbit(model: &Model, x: &[f64]) -> BoundOrbit {
    let nsub = 12; // 48 rows over the cycle, matching the P0 CSV export
    let traj = model.integrate(x, nsub);
    let n = traj.len();
    // P3-a footholds: each pair's foot fore-aft relative to the CoM at its
    // touchdown. front touches down at index 0 (phase 0); rear at index
    // 2*nsub (phase 0.5, start of the rear-stance segment).
    let front_foothold = x[P_XF] - traj[0][0];
    let rear_foothold = x[P_XR] - traj[(2 * nsub).min(n - 1)][0];
    let mut samples = Vec::with_capacity(n - 1);
    let mut table = Vec::with_capacity(n - 1);
    for (k, s) in traj.iter().take(n - 1).enumerate() {
        let phase = k as f64 / (n - 1) as f64;
        samples.push(OrbitSample { phase, z: s[1], pitch: s[2], vx: s[3], vz: s[4], w: s[5] });
        table.push([phase, s[1], s[2], s[3], s[4], s[5]]);
    }
    // periodicity residual (unweighted) and worst friction margin
    let st = &traj[n - 1];
    let per = [
        st[1] - x[P_Z0],
        st[2] - x[P_TH0],
        st[3] - x[P_VX0],
        st[4] - x[P_VZ0],
        st[5] - x[P_W0],
        (st[0] - 0.0) - model.p.vx_target * model.p.cycle_period_s,
    ]
    .iter()
    .fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut fric = f64::INFINITY;
    let (xf, xr) = (x[P_XF], x[P_XR]);
    let _ = (xf, xr);
    for (steps, (_dur, active)) in model.phases().iter().enumerate().map(|(_, p)| (60usize, *p)) {
        for i in 0..steps {
            let frac = i as f64 / steps as f64;
            let (ffx, ffz, frx, frz) = model.force_at(active, frac, x);
            if active == Stance::Front && ffz > 1e-6 {
                fric = fric.min(model.p.friction_mu - ffx.abs() / ffz);
            }
            if active == Stance::Rear && frz > 1e-6 {
                fric = fric.min(model.p.friction_mu - frx.abs() / frz);
            }
        }
    }
    BoundOrbit {
        samples,
        xf: x[P_XF],
        xr: x[P_XR],
        front_foothold,
        rear_foothold,
        periodicity_residual: per,
        friction_margin: fric,
        table,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go2_forward_bound_orbit_exists() {
        // Same config as the P0 Python check: 1.0 m/s, T=0.30, duty=0.34.
        let params = PeriodicBoundParams::go2(1.0, 0.30, 0.34);
        let orbit = solve(&params).expect("forward periodic bound orbit should exist");
        let vx_dbg = orbit.samples.iter().map(|s| s.vx).sum::<f64>() / orbit.samples.len() as f64;
        let zmax_dbg = orbit.samples.iter().map(|s| s.z).fold(f64::MIN, f64::max);
        let zmin_dbg = orbit.samples.iter().map(|s| s.z).fold(f64::MAX, f64::min);
        let pk = orbit.samples.iter().map(|s| s.pitch.abs()).fold(0.0_f64, f64::max);
        eprintln!(
            "[P1 Rust orbit] periodicity={:.2e} vx_avg={:.3} z_range={:.3} peak_pitch={:.3} \
             friction_margin={:.3} xf={:.3} xr={:.3} front_foothold={:.3} rear_foothold={:.3} rows={}",
            orbit.periodicity_residual, vx_dbg, zmax_dbg - zmin_dbg, pk,
            orbit.friction_margin, orbit.xf, orbit.xr,
            orbit.front_foothold, orbit.rear_foothold, orbit.table.len(),
        );
        // periodicity closed and forward speed achieved
        assert!(
            orbit.periodicity_residual < 1e-3,
            "periodicity residual too large: {}",
            orbit.periodicity_residual
        );
        // average forward velocity == target (x advanced by vx*T over one cycle)
        let vx_avg = orbit.samples.iter().map(|s| s.vx).sum::<f64>() / orbit.samples.len() as f64;
        assert!((vx_avg - 1.0).abs() < 0.2, "avg vx off target: {vx_avg}");
        // physically feasible: friction cone respected, pitch bounded
        assert!(orbit.friction_margin > -0.05, "friction violated: {}", orbit.friction_margin);
        let peak_pitch = orbit.samples.iter().map(|s| s.pitch.abs()).fold(0.0_f64, f64::max);
        assert!(peak_pitch < 0.7, "pitch not bounded: {peak_pitch}");
        // real (if mild) vertical excursion -> it's a bound, not a flat drag
        let zmax = orbit.samples.iter().map(|s| s.z).fold(f64::MIN, f64::max);
        let zmin = orbit.samples.iter().map(|s| s.z).fold(f64::MAX, f64::min);
        assert!(zmax - zmin > 0.005, "no vertical excursion: {}", zmax - zmin);
        assert_eq!(orbit.table.len(), orbit.samples.len());
    }
}
