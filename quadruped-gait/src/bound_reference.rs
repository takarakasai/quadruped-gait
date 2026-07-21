//! Closed-form periodic "trim" reference for the Bound gait's
//! front-pair/rear-pair stance.
//!
//! Derived from a single-rigid-body (SRBD) periodic boundary-value
//! problem: front-pair stance occupies cycle phase `[0, 0.5)`,
//! rear-pair stance occupies `[0.5, 1.0)` (matching
//! [`crate::config::GaitType::Bound`]'s own phase offsets,
//! `FL/FR=0.0, RL/RR=0.5`), with `duty_factor=0.5` -- i.e. no aerial
//! phase, the two pairs tile the cycle with zero gap.
//!
//! Both feet of a stance pair share the same body-frame `r_x` moment
//! arm (front pair: both feet at `+r_x`; rear pair: both at `-r_x`),
//! so splitting `F_z` between them buys zero pitch torque -- pitch
//! authority comes entirely from the pair's shared fore-aft force
//! `F_x` (`τ = -h0·F_x - r_x·F_z`). Exploiting front/rear mirror
//! symmetry (`F_x` negates, `F_z` doesn't, between the two phases)
//! reduces the problem to a single half-cycle boundary-value problem
//! with piecewise-constant `F_x`/`F_z` per phase, solved in closed
//! form below. See `articara/ref/wbc_comparison.md` Sec.5bb (local)
//! for the full derivation and the real-Go2 numeric evaluation.

const GRAVITY_MPS2: f64 = 9.81;

/// Physical + gait-timing parameters the trim model needs. Populate
/// from real, auto-detected values (`auto_detect_srbd_mpc_config`'s
/// `mass_kg`/`inertia_diag_body.y`, `auto_detect_kinematics_config`'s
/// `nominal_foot_body.x`/`.z`) -- never hand-typed constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundTrimConfig {
    pub mass_kg: f64,
    /// Body-frame pitch-axis inertia (`inertia_diag_body.y`).
    pub inertia_yy: f64,
    /// Body-frame fore-aft moment arm magnitude from CoM to a stance
    /// foot (front and rear are assumed symmetric; use their average
    /// if they differ slightly, as real geometry usually does).
    pub r_x: f64,
    /// Nominal standing height (magnitude of the foot's body-frame
    /// vertical offset from the CoM).
    pub h0: f64,
    pub cycle_period_s: f64,
    pub duty_factor: f64,
    /// Friction-cone coefficient the trim's `F_x` is clipped against.
    /// Pass the WBC/MPC's own `friction_mu` belief to keep the
    /// reference self-consistent with whatever the solver is
    /// configured with.
    pub friction_mu: f64,
    /// Global sign multiplier (`+1.0` or `-1.0`) applied to every
    /// `sample()` output (`pitch`, `pitch_rate`, `f_x_per_leg`).
    /// This module's own derivation fixes a "front stance ⇒ positive
    /// phase-A pitch sign" convention internally, self-consistent by
    /// construction (see the unit tests below) -- but that internal
    /// convention isn't guaranteed to match whichever sign the
    /// consuming pipeline's own `base_euler_zyx.y` / `euler_angles()`
    /// happens to use for "positive pitch". Rather than guessing,
    /// this field makes the choice a single, explicit, empirically-
    /// checkable knob: a MuJoCo phase-check (comparing this model's
    /// `sample().pitch` against the real measured pitch over a cycle)
    /// found them running near-antiphase at `sign=1.0` -- consistent
    /// with a sign mismatch, not just a tracking lag -- so `sign=-1.0`
    /// is the empirically-corrected value for Go2's real convention
    /// (see `articara/ref/wbc_comparison.md` Sec.5bc, local doc).
    pub sign: f64,
}

/// The trim reference at one instant: pitch/pitch-rate and the
/// per-leg force split for whichever pair is currently in stance
/// (the caller determines which pair from the same `cycle_phase`
/// used to call [`BoundTrimConfig::sample`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundTrimSample {
    pub pitch: f64,
    pub pitch_rate: f64,
    /// Fore-aft GRF for ONE foot of the currently-stance pair (total
    /// pair force / 2).
    pub f_x_per_leg: f64,
    /// Vertical GRF for ONE foot of the currently-stance pair (total
    /// pair force / 2, `= m·g/2` always -- see [`Self::f_z_total`]).
    pub f_z_per_leg: f64,
}

impl BoundTrimConfig {
    /// Total vertical GRF (both stance feet). Constant `= m·g`
    /// throughout the cycle -- height closure (`ż` periodic under a
    /// constant `F_z`) forces this; Bound at `duty_factor=0.5` has no
    /// aerial phase, so no vertical bounce is kinematically possible
    /// or required.
    pub fn f_z_total(&self) -> f64 {
        self.mass_kg * GRAVITY_MPS2
    }

    /// The fore-aft force (both stance feet) that exactly zeroes net
    /// pitch torque: `F_x* = -r_x·m·g/h0`, from `τ = -h0·F_x -
    /// r_x·F_z = 0` at `F_z = m·g`.
    pub fn f_x_trim(&self) -> f64 {
        -self.r_x * self.mass_kg * GRAVITY_MPS2 / self.h0
    }

    /// Friction coefficient required to realize the *exact* trim
    /// (`|F_x*| / (m·g) = r_x/h0`), independent of `friction_mu`.
    pub fn mu_needed(&self) -> f64 {
        self.f_x_trim().abs() / self.f_z_total()
    }

    /// `f_x_trim()` clipped to the friction cone at `self.friction_mu`
    /// -- the actually-realizable fore-aft force when the exact trim
    /// exceeds the available friction budget.
    pub fn f_x_clipped(&self) -> f64 {
        let bound = self.friction_mu * self.f_z_total();
        self.f_x_trim().clamp(-bound, bound)
    }

    fn t_st(&self) -> f64 {
        self.cycle_period_s * self.duty_factor
    }

    /// Angular acceleration during front-pair stance for a given
    /// (front-phase-convention) fore-aft force `f_x`: `α_p = (-h0·f_x
    /// - r_x·F_z) / I_yy`.
    fn alpha_p(&self, f_x: f64) -> f64 {
        (-self.h0 * f_x - self.r_x * self.f_z_total()) / self.inertia_yy
    }

    /// Peak pitch magnitude (rad) reached at mid-stance, for a given
    /// front-phase fore-aft force `f_x`: `|α_p|·T_st²/8`.
    pub fn theta_peak(&self, f_x: f64) -> f64 {
        (self.alpha_p(f_x).abs() * self.t_st() * self.t_st()) / 8.0
    }

    /// Sample the trim reference at a global gait-cycle phase
    /// `cycle_phase ∈ [0, 1)` (matching `PhaseGenerator::cycle_phase()`
    /// / `GaitType::Bound`'s own front=`[0,0.5)`/rear=`[0.5,1.0)`
    /// convention). Uses the closed-form solution to the half-cycle
    /// boundary-value problem: `θ_A(s) = (α_p/2)·s·(s−T_st)` for front
    /// stance (`θ_A(0)=0`, peak magnitude at `s=T_st/2`), and rear
    /// stance is the front solution's exact negation at the same
    /// local stance time (`θ_B(s) = -θ_A(s)`, `F_x^B = -F_x^A`) --
    /// the mirror-symmetry ansatz that closes the periodicity
    /// condition (see module docs / Sec.5bb).
    pub fn sample(&self, cycle_phase: f64) -> BoundTrimSample {
        let cycle_phase = cycle_phase.rem_euclid(1.0);
        let front_stance = cycle_phase < 0.5;
        let local_frac = if front_stance { cycle_phase } else { cycle_phase - 0.5 }; // in [0, 0.5)
        let s = local_frac * 2.0 * self.t_st(); // map [0,0.5) -> [0, T_st)

        let f_x_a = self.f_x_clipped();
        let alpha_p = self.alpha_p(f_x_a);
        let t_st = self.t_st();
        let theta_a = (alpha_p / 2.0) * s * (s - t_st);
        let theta_dot_a = alpha_p * (s - t_st / 2.0);

        let (pitch, pitch_rate, f_x_pair) = if front_stance {
            (theta_a, theta_dot_a, f_x_a)
        } else {
            (-theta_a, -theta_dot_a, -f_x_a)
        };

        BoundTrimSample {
            pitch: self.sign * pitch,
            pitch_rate: self.sign * pitch_rate,
            f_x_per_leg: self.sign * f_x_pair / 2.0,
            f_z_per_leg: self.f_z_total() / 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real Go2 numbers (from Sec.5bb / `go2_diag_bound_trim_model_
    /// feasibility` in articara's test suite), so these tests double
    /// as a regression check against that diagnostic's printed
    /// values.
    fn go2_cfg(friction_mu: f64) -> BoundTrimConfig {
        BoundTrimConfig {
            mass_kg: 15.606,
            inertia_yy: 0.0981,
            r_x: 0.1922,
            h0: 0.2664,
            cycle_period_s: 0.30,
            duty_factor: 0.5,
            friction_mu,
            sign: 1.0,
        }
    }

    #[test]
    fn f_x_trim_and_mu_needed_match_hand_derivation() {
        let cfg = go2_cfg(0.7);
        assert!((cfg.f_x_trim() - (-110.44)).abs() < 0.1);
        assert!((cfg.mu_needed() - 0.721).abs() < 0.01);
        assert!((cfg.f_z_total() - 153.10).abs() < 0.1);
    }

    #[test]
    fn theta_peak_matches_hand_derivation() {
        let cfg = go2_cfg(0.7);
        assert!((cfg.theta_peak(0.0) - 0.8436).abs() < 0.01);
        assert!(cfg.theta_peak(cfg.f_x_trim()) < 1e-9);
        assert!((cfg.theta_peak(cfg.f_x_clipped()) - 0.0250).abs() < 0.005);
    }

    #[test]
    fn pitch_is_zero_at_both_pair_switch_instants() {
        let cfg = go2_cfg(0.7);
        assert!(cfg.sample(0.0).pitch.abs() < 1e-9);
        assert!(cfg.sample(0.5).pitch.abs() < 1e-9);
    }

    #[test]
    fn pitch_peak_magnitude_at_mid_stance_matches_theta_peak() {
        let cfg = go2_cfg(0.7);
        let mid_front = cfg.sample(0.25).pitch.abs();
        let mid_rear = cfg.sample(0.75).pitch.abs();
        let expected = cfg.theta_peak(cfg.f_x_clipped());
        assert!((mid_front - expected).abs() < 1e-6, "front mid-stance: {mid_front} vs {expected}");
        assert!((mid_rear - expected).abs() < 1e-6, "rear mid-stance: {mid_rear} vs {expected}");
    }

    #[test]
    fn rear_phase_is_exact_mirror_of_front_phase() {
        let cfg = go2_cfg(0.7);
        for i in 0..50 {
            let frac = i as f64 / 100.0; // [0, 0.5)
            let front = cfg.sample(frac);
            let rear = cfg.sample(frac + 0.5);
            assert!((front.pitch + rear.pitch).abs() < 1e-9);
            assert!((front.pitch_rate + rear.pitch_rate).abs() < 1e-9);
            assert!((front.f_x_per_leg + rear.f_x_per_leg).abs() < 1e-9);
            assert!((front.f_z_per_leg - rear.f_z_per_leg).abs() < 1e-9);
        }
    }

    #[test]
    fn f_z_per_leg_is_half_body_weight_always() {
        let cfg = go2_cfg(0.7);
        for i in 0..10 {
            let phase = i as f64 / 10.0;
            let expected = cfg.f_z_total() / 2.0;
            assert!((cfg.sample(phase).f_z_per_leg - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn tighter_friction_reduces_available_thrust_and_raises_peak_pitch() {
        let loose = go2_cfg(1.5);
        let tight = go2_cfg(0.5);
        assert!(loose.f_x_clipped().abs() >= tight.f_x_clipped().abs());
        assert!(loose.theta_peak(loose.f_x_clipped()) <= tight.theta_peak(tight.f_x_clipped()));
    }
}
