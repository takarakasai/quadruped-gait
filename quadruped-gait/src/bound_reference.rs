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
    /// Deliberate fraction (`[0,1]`) of the friction-clipped trim
    /// force actually commanded: `F_x_used = thrust_scale *
    /// f_x_clipped()`. `1.0` (the default/prior behaviour) uses the
    /// full clipped trim, which at Go2's real numbers already
    /// saturates the hard friction cone by itself
    /// (`mu_needed=0.721 > friction_mu=0.7`) and leaves zero headroom
    /// for the MPC/WBC's own velocity-tracking `F_x` -- confirmed as
    /// the root cause of cmd_vx's near-zero effect on `meas_vx` in
    /// `articara/ref/wbc_comparison.md` Sec.5bf. Values `<1.0`
    /// deliberately under-cancel the pitch torque, trading a larger
    /// `theta_peak` for real, physically-honest friction headroom
    /// (`(1-thrust_scale)*mu*F_z` per pair) the rest of the pipeline
    /// can actually spend on velocity tracking without violating the
    /// friction cone (see the handover memo's Sec.6(c) "ピッチの許容量
    /// を増やす" option, and `ref/scripts/
    /// simulate_point_mass_bound_sweep.py`'s partial-trim sweep).
    pub thrust_scale: f64,
    /// Current commanded forward speed (m/s). Only consulted when
    /// [`Self::velocity_ripple_fraction`] is `Some`; ignored (may be
    /// left `0.0`) on the default `thrust_scale`-driven path.
    pub cmd_vx_mps: f64,
    /// If `Some(fraction)`, [`Self::f_x_used`] is instead sized from a
    /// TARGET peak-to-peak velocity ripple (`fraction * |cmd_vx_mps|`)
    /// via the same `delta_v = |F_x|/m·T_st` relation (inverted),
    /// friction-clipped -- the "impulse scaling" alternative to
    /// `thrust_scale`'s pitch-cancellation-then-derate approach (MIT
    /// Cheetah 2's vertical/horizontal impulse scaling, Park/Wensing/
    /// Kim 2017; see also Poulakakis/Papadopoulos/Buehler's Scout II
    /// passive-dynamics analysis and Cheng/Alqaham/Gan 2024's
    /// "Harnessing Natural Oscillations", all of which size stance
    /// force from the desired velocity change first and treat pitch as
    /// a resulting, monitored-not-minimized side effect). `theta_peak`
    /// is no longer a design target on this path -- read it back after
    /// the fact via [`Self::theta_peak`]. `None` (default) preserves
    /// the `thrust_scale`-based behaviour exactly (`articara/ref/
    /// wbc_comparison.md` Sec.5bj, local doc, has the full derivation
    /// and the empirical calibration against `thrust_scale=0.4`).
    pub velocity_ripple_fraction: Option<f64>,
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
    /// pair force / 2, `= m·g/2` at `duty_factor=0.5`; `0.0` during a
    /// flight-phase sample when `duty_factor<0.5` -- see
    /// [`Self::f_z_total`]).
    pub f_z_per_leg: f64,
    /// Reference **vertical CoM velocity** (m/s, world +z = up) of the
    /// ballistic bounce this trim implies (Sec.5d4, local doc). Zero at
    /// `duty_factor=0.5` (no flight, no bounce); for `duty<0.5` it is
    /// the closed-form bounce velocity: during stance the CoM
    /// accelerates upward at `a = g·T_flight/T_st` from the touchdown
    /// speed `−g·T_flight/2` to the liftoff speed `+g·T_flight/2`;
    /// during flight it decelerates ballistically at `−g`. Feeding this
    /// (with the `f_z` surplus) into the MPC's vertical reference makes
    /// the reference physically FEASIBLE through the aerial phase --
    /// the flat `v_z=0` reference otherwise fights the bounce force.
    pub com_z_velocity: f64,
}

impl BoundTrimConfig {
    /// Total vertical GRF during stance (both stance feet). `= m·g`
    /// at `duty_factor=0.5` (Bound's original, no-aerial-phase case,
    /// where `ż` closes periodically under a constant `F_z` with no
    /// bounce needed or possible). Generalizes to `duty_factor<0.5`
    /// (a genuine aerial phase between pairs, see module docs): flight
    /// is ballistic (`F_z=0`), so the parabola closes in exactly
    /// `T_flight` seconds only if liftoff/landing vertical speed is
    /// `+-g·T_flight/2`; stance must then supply the vertical impulse
    /// `m·g·T_flight` that reverses that speed, over duration `T_st`
    /// -- averaging (piecewise-constant, matching this model's
    /// existing philosophy for `F_x`) gives `F_z = m·g/(2·duty_
    /// factor)`, which is exactly `m·g` at `duty_factor=0.5`.
    /// Duty as the trim's own closed form may use it, clamped to 0.5.
    ///
    /// The derivation assumes the two pairs TILE the cycle: exactly one pair
    /// loaded at a time, so `F_z (2 d) = m g` and the half-cycle splits as
    /// `[0, T_st)` then flight. Both break for `d > 0.5`, which is the
    /// overlap (4-support) regime:
    ///   - `f_z_total` would return `m g / (2 d) < m g`, under-supporting
    ///     gravity by 9% at d = 0.55 and shrinking `f_x_trim` by the same
    ///     factor.
    ///   - `t_st = T d` would exceed `T/2`, so `sample()`'s half-cycle
    ///     partition overruns and the periodicity closure `theta(T/2) =
    ///     -theta(0)` no longer holds -- leaving a step discontinuity in the
    ///     pitch reference at every half-cycle boundary.
    /// Clamping here keeps the reference well-posed while letting the CONTACT
    /// SCHEDULE use the real `duty_factor`, which is the point of running
    /// `d > 0.5`: genuine 4-support rows where both pairs share `F_z`, the
    /// `r_x` moment arms cancel, and pitch needs no friction at all.
    ///
    /// Strictly a no-op for every `duty_factor <= 0.5`, so no existing result
    /// moves.
    fn duty_trim(&self) -> f64 {
        self.duty_factor.min(0.5)
    }

    pub fn f_z_total(&self) -> f64 {
        self.mass_kg * GRAVITY_MPS2 / (2.0 * self.duty_trim())
    }

    /// The fore-aft force (both stance feet) that exactly zeroes net
    /// pitch torque: `F_x* = -r_x·F_z/h0`, from `τ = -h0·F_x -
    /// r_x·F_z = 0`. Uses [`Self::f_z_total`] (not a hardcoded `m·g`)
    /// so the `duty_factor<0.5` dependence propagates here too --
    /// equal to `-r_x·m·g/h0` at `duty_factor=0.5`.
    pub fn f_x_trim(&self) -> f64 {
        -self.r_x * self.f_z_total() / self.h0
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

    /// The fore-aft force actually commanded by [`Self::sample`].
    /// When [`Self::velocity_ripple_fraction`] is `Some(fraction)`,
    /// sized from the target velocity ripple (friction-clipped,
    /// signed to match [`Self::f_x_trim`]'s sign convention --
    /// `alpha_p` isn't symmetric in the sign of `f_x`, so getting the
    /// sign right matters for the resulting `theta_peak`). Otherwise
    /// (`None`, the default), `f_x_clipped()` scaled by
    /// [`Self::thrust_scale`] -- equal to `f_x_clipped()` itself at
    /// `thrust_scale=1.0`.
    pub fn f_x_used(&self) -> f64 {
        match self.velocity_ripple_fraction {
            Some(fraction) => {
                let ripple_pp = fraction * self.cmd_vx_mps.abs();
                let bound = self.friction_mu * self.f_z_total();
                let f_x_mag = (self.mass_kg * ripple_pp / self.t_st()).min(bound);
                f_x_mag * self.f_x_trim().signum()
            }
            None => self.thrust_scale * self.f_x_clipped(),
        }
    }

    fn t_st(&self) -> f64 {
        self.cycle_period_s * self.duty_trim()
    }

    /// Aerial (flight) duration within a half-cycle -- `0.0` at
    /// `duty_factor=0.5` (Bound's original, no-aerial-phase case);
    /// positive whenever `duty_factor<0.5` opens a genuine moment
    /// with all 4 legs in swing (see module docs).
    fn t_flight(&self) -> f64 {
        (0.5 - self.duty_factor).max(0.0) * self.cycle_period_s
    }

    /// Angular acceleration during front-pair stance for a given
    /// (front-phase-convention) fore-aft force `f_x`: `α_p = (-h0·f_x
    /// - r_x·F_z) / I_yy`. Only applies during stance -- flight is
    /// force-free (`θ̈=0`), handled separately in [`Self::sample`].
    fn alpha_p(&self, f_x: f64) -> f64 {
        (-self.h0 * f_x - self.r_x * self.f_z_total()) / self.inertia_yy
    }

    /// Pitch and pitch-rate at front-pair touchdown (`s=0`), from the
    /// closed-form periodicity closure across stance+flight: the
    /// front-stance trajectory, propagated through the following
    /// flight segment, must land on `(-θ(0), -θ̇(0))` at the half-cycle
    /// mark `s=T/2` for the mirror ansatz to close (see module docs).
    /// `θ̇(0) = -α_p·T_st/2` and `θ(0) = -α_p·T_st·T_flight/4` --
    /// `T_flight=0` (`duty_factor=0.5`) gives `θ(0)=0` exactly,
    /// matching the original derivation's boundary condition.
    fn theta_boundary(&self, f_x: f64) -> (f64, f64) {
        let t_st = self.t_st();
        let t_flight = self.t_flight();
        let a = self.alpha_p(f_x);
        let theta_dot_0 = -a * t_st / 2.0;
        let theta_0 = -a * t_st * t_flight / 4.0;
        (theta_0, theta_dot_0)
    }

    /// Peak pitch magnitude (rad) reached over the half-cycle, for a
    /// given front-phase fore-aft force `f_x`. Candidates: `s=0`
    /// (`|θ(0)|`, generally nonzero once a flight phase is present),
    /// the interior stance extremum (where `θ̇=0`, if it falls within
    /// `[0,T_st]`), and `s=T_st` (liftoff -- flight is linear in `s`,
    /// so its own extrema are its endpoints, already covered by
    /// `s=T_st` and the `s=T/2` closure value `-θ(0)`). At
    /// `duty_factor=0.5` this reduces to the original single-parabola
    /// peak (`θ(0)=0`, extremum at `s=T_st/2`).
    pub fn theta_peak(&self, f_x: f64) -> f64 {
        let t_st = self.t_st();
        let a = self.alpha_p(f_x);
        let (theta_0, theta_dot_0) = self.theta_boundary(f_x);
        let theta_at = |s: f64| theta_0 + theta_dot_0 * s + 0.5 * a * s * s;
        let mut peak = theta_0.abs().max(theta_at(t_st).abs());
        if a != 0.0 {
            let s_star = -theta_dot_0 / a;
            if s_star > 0.0 && s_star < t_st {
                peak = peak.max(theta_at(s_star).abs());
            }
        }
        peak
    }

    /// Sample the trim reference at a global gait-cycle phase
    /// `cycle_phase ∈ [0, 1)` (matching `PhaseGenerator::cycle_phase()`
    /// / `GaitType::Bound`'s own front=`[0,0.5)`/rear=`[0.5,1.0)`
    /// convention). Each half-cycle is now `[0,T_st)` stance followed
    /// by `[T_st,T/2)` flight (T_flight=0 at `duty_factor=0.5`,
    /// reducing exactly to the original single-segment closed form):
    /// stance follows the quadratic from [`Self::theta_boundary`]'s
    /// initial conditions, flight continues at the constant angular
    /// velocity reached at liftoff (force-free) with `f_x_per_leg=
    /// f_z_per_leg=0`. Rear stance is the front solution's exact
    /// negation at the same local half-cycle time (`θ_B(s)=-θ_A(s)`,
    /// `F_x^B=-F_x^A`; `F_z` does not flip sign, it's the same
    /// stance/flight schedule shape for both pairs, just shifted) --
    /// the mirror-symmetry ansatz that closes the periodicity
    /// condition (see module docs / Sec.5bb/5bq).
    pub fn sample(&self, cycle_phase: f64) -> BoundTrimSample {
        let cycle_phase = cycle_phase.rem_euclid(1.0);
        let front_half = cycle_phase < 0.5;
        let local_frac = if front_half { cycle_phase } else { cycle_phase - 0.5 }; // in [0, 0.5)
        let s = local_frac * self.cycle_period_s; // local half-cycle time, in [0, T/2)

        let f_x_a = self.f_x_used();
        let alpha_p = self.alpha_p(f_x_a);
        let t_st = self.t_st();
        let (theta_0, theta_dot_0) = self.theta_boundary(f_x_a);

        let (theta_a, theta_dot_a, f_x_pair_a, f_z_pair_a) = if s < t_st {
            // Stance: quadratic from the touchdown boundary condition.
            let theta = theta_0 + theta_dot_0 * s + 0.5 * alpha_p * s * s;
            let theta_dot = theta_dot_0 + alpha_p * s;
            (theta, theta_dot, f_x_a, self.f_z_total())
        } else {
            // Flight: force-free, constant angular velocity from liftoff.
            let theta_dot_lo = theta_dot_0 + alpha_p * t_st;
            let theta_lo = theta_0 + theta_dot_0 * t_st + 0.5 * alpha_p * t_st * t_st;
            let u = s - t_st;
            (theta_lo + theta_dot_lo * u, theta_dot_lo, 0.0, 0.0)
        };

        let (pitch, pitch_rate, f_x_pair, f_z_pair) = if front_half {
            (theta_a, theta_dot_a, f_x_pair_a, f_z_pair_a)
        } else {
            (-theta_a, -theta_dot_a, -f_x_pair_a, f_z_pair_a)
        };

        // Ballistic vertical bounce velocity (Sec.5d4). Same shape in
        // both half-cycles (period T/2), so it's a function of the
        // local half-cycle time `s` only -- no front/rear sign flip
        // (up is up). Zero when `t_flight == 0` (duty_factor>=0.5).
        let t_flight = self.t_flight();
        let com_z_velocity = if t_flight > 0.0 && t_st > 0.0 {
            let v_liftoff = 0.5 * GRAVITY_MPS2 * t_flight; // = g·T_flight/2
            if s < t_st {
                // Stance: from touchdown (−v_liftoff) accelerating up at
                // a = g·T_flight/T_st to liftoff (+v_liftoff).
                let a_stance = GRAVITY_MPS2 * t_flight / t_st;
                -v_liftoff + a_stance * s
            } else {
                // Flight: ballistic decel at −g from +v_liftoff.
                v_liftoff - GRAVITY_MPS2 * (s - t_st)
            }
        } else {
            0.0
        };

        BoundTrimSample {
            pitch: self.sign * pitch,
            pitch_rate: self.sign * pitch_rate,
            f_x_per_leg: self.sign * f_x_pair / 2.0,
            f_z_per_leg: f_z_pair / 2.0,
            com_z_velocity,
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
            thrust_scale: 1.0,
            cmd_vx_mps: 0.0,
            velocity_ripple_fraction: None,
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

    #[test]
    fn thrust_scale_one_reproduces_prior_behaviour() {
        let cfg = go2_cfg(0.7);
        assert!((cfg.f_x_used() - cfg.f_x_clipped()).abs() < 1e-9);
    }

    #[test]
    fn thrust_scale_zero_means_zero_commanded_force_and_theta_peak_matches_zero_fx() {
        let mut cfg = go2_cfg(0.7);
        cfg.thrust_scale = 0.0;
        assert!((cfg.f_x_used()).abs() < 1e-9);
        let mid_front = cfg.sample(0.25).pitch.abs();
        assert!((mid_front - cfg.theta_peak(0.0)).abs() < 1e-6);
    }

    #[test]
    fn theta_peak_grows_monotonically_as_thrust_scale_shrinks() {
        let mut cfg = go2_cfg(0.7);
        let mut prev_theta_peak = cfg.theta_peak(cfg.f_x_used());
        for thrust_scale in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] {
            cfg.thrust_scale = thrust_scale;
            let theta_peak = cfg.theta_peak(cfg.f_x_used());
            assert!(
                theta_peak >= prev_theta_peak,
                "thrust_scale={thrust_scale}: theta_peak {theta_peak} should be >= previous {prev_theta_peak}"
            );
            prev_theta_peak = theta_peak;
        }
    }

    #[test]
    fn f_x_used_scales_linearly_with_thrust_scale() {
        let mut cfg = go2_cfg(0.7);
        let full = cfg.f_x_clipped();
        for thrust_scale in [0.0, 0.25, 0.5, 0.75, 1.0] {
            cfg.thrust_scale = thrust_scale;
            assert!((cfg.f_x_used() - thrust_scale * full).abs() < 1e-9);
        }
    }

    /// Sec.5bj: `velocity_ripple_fraction=None` must reproduce the
    /// `thrust_scale` path exactly, whatever `thrust_scale` is set to
    /// -- this is what makes the new field purely additive.
    #[test]
    fn velocity_ripple_fraction_none_reproduces_thrust_scale_behaviour() {
        let mut cfg = go2_cfg(0.7);
        for thrust_scale in [0.0, 0.4, 1.0] {
            cfg.thrust_scale = thrust_scale;
            assert!((cfg.f_x_used() - thrust_scale * cfg.f_x_clipped()).abs() < 1e-9);
        }
    }

    /// Unclipped branch: `|F_x| = mass_kg * fraction * |cmd_vx| /
    /// T_st`, matching `ref/scripts/simulate_point_mass_bound_sweep.py`'s
    /// `f_x_from_ripple_fraction`. Uses a small fraction/cmd_vx so the
    /// friction cone never binds, isolating the linear relation.
    #[test]
    fn velocity_ripple_unclipped_matches_closed_form() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        cfg.cmd_vx_mps = 0.40;
        cfg.velocity_ripple_fraction = Some(0.3);
        let t_st = cfg.cycle_period_s * cfg.duty_factor;
        let expected_mag = cfg.mass_kg * 0.3 * 0.40 / t_st;
        assert!(expected_mag < cfg.friction_mu * cfg.f_z_total(), "test setup should stay unclipped");
        assert!((cfg.f_x_used().abs() - expected_mag).abs() < 1e-6);
    }

    /// Sign convention: `f_x_used()` on the velocity-ripple path must
    /// carry the same sign as `f_x_trim()` (both are the front-phase
    /// force), not an arbitrary/positive magnitude -- `alpha_p` isn't
    /// symmetric in the sign of `f_x` (Sec.5bj's sign-bug fix in the
    /// Python sweep script came from exactly this).
    #[test]
    fn velocity_ripple_f_x_used_sign_matches_f_x_trim() {
        let mut cfg = go2_cfg(0.7);
        cfg.cmd_vx_mps = 0.40;
        cfg.velocity_ripple_fraction = Some(0.3);
        assert_eq!(cfg.f_x_used().signum(), cfg.f_x_trim().signum());
    }

    /// Clipped branch: at a high enough cmd_vx/fraction, `f_x_used()`
    /// saturates at the friction cone (`mu*F_z`) exactly like
    /// `f_x_clipped()` does on the `thrust_scale` path -- the two
    /// paths share the same physical ceiling.
    #[test]
    fn velocity_ripple_clips_to_friction_cone_at_high_cmd_vx() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        cfg.cmd_vx_mps = 5.0; // deliberately large -> must saturate
        cfg.velocity_ripple_fraction = Some(1.0);
        let bound = cfg.friction_mu * cfg.f_z_total();
        assert!((cfg.f_x_used().abs() - bound).abs() < 1e-9);
    }

    /// `theta_peak` at the resulting `f_x_used()` grows as `cmd_vx`
    /// shrinks (a smaller commanded speed needs less `F_x` to hit the
    /// same ripple *fraction*, moving further from the pitch-canceling
    /// `f_x_trim()`) -- confirms pitch really is just a readout here,
    /// not a target, and that the readout responds sensibly to cmd_vx.
    #[test]
    fn velocity_ripple_theta_peak_shrinks_as_cmd_vx_grows_toward_full_trim() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        cfg.velocity_ripple_fraction = Some(0.6);
        let mut prev_theta_peak = f64::INFINITY;
        for cmd_vx in [0.2, 0.4, 0.8, 1.2] {
            cfg.cmd_vx_mps = cmd_vx;
            let theta_peak = cfg.theta_peak(cfg.f_x_used());
            assert!(
                theta_peak <= prev_theta_peak,
                "cmd_vx={cmd_vx}: theta_peak {theta_peak} should be <= previous {prev_theta_peak}"
            );
            prev_theta_peak = theta_peak;
        }
    }

    /// Calibration check against Sec.5bg's hand-found best empirical
    /// point (`thrust_scale=0.4` at T=0.18/cmd_vx=0.40 gave
    /// `F_x_used=-42.87N`): `velocity_ripple_fraction≈0.62` should
    /// reproduce essentially the same `F_x_used` from cmd_vx alone,
    /// per `simulate_point_mass_bound_sweep.py`'s own sanity check.
    #[test]
    fn velocity_ripple_fraction_calibrates_against_thrust_scale_0_4_empirical_point() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        let thrust_scale_path = {
            let mut c = cfg;
            c.thrust_scale = 0.4;
            c.f_x_used()
        };
        cfg.cmd_vx_mps = 0.40;
        cfg.velocity_ripple_fraction = Some(0.62);
        assert!(
            (cfg.f_x_used() - thrust_scale_path).abs() < 2.0,
            "ripple-fraction F_x_used={:.2} should be within 2N of thrust_scale=0.4's {:.2}",
            cfg.f_x_used(), thrust_scale_path
        );
    }

    /// `f_z_total()` at `duty_factor<0.5` matches the flight-phase
    /// closed form (`m·g/(2·duty)`), cross-checked against
    /// `simulate_point_mass_bound_flight_phase.py`'s printed table
    /// (Phase 0, local doc Sec.5bq).
    #[test]
    fn duty_above_half_is_clamped_and_leaves_lower_duties_untouched() {
        // The clamp must be a strict no-op at duty <= 0.5 -- every recorded
        // result in this repo depends on that -- and must keep the closed form
        // well-posed above it.
        let mk = |d: f64| BoundTrimConfig { duty_factor: d, ..go2_cfg(0.7) };
        for d in [0.25, 0.34, 0.40, 0.50] {
            let c = mk(d);
            let expected = c.mass_kg * GRAVITY_MPS2 / (2.0 * d);
            assert!((c.f_z_total() - expected).abs() < 1e-9,
                    "duty {d} must be untouched by the clamp");
        }
        // Above 0.5 everything pins to the duty-0.5 value rather than drifting
        // into the under-supporting / closure-breaking regime.
        let at_half = mk(0.50);
        for d in [0.55, 0.60, 0.70] {
            let c = mk(d);
            assert!((c.f_z_total() - at_half.f_z_total()).abs() < 1e-9,
                    "duty {d} must clamp to the duty-0.5 support force");
            // gravity is fully supported, never the 0.909*m*g the unclamped
            // form would give at d = 0.55
            assert!(c.f_z_total() >= c.mass_kg * GRAVITY_MPS2 - 1e-9,
                    "duty {d} must not under-support gravity");
        }
    }

    #[test]
    fn duty_above_half_keeps_the_half_cycle_pitch_reference_continuous() {
        // sample() partitions the half-cycle as [0, t_st) then flight. With
        // t_st > T/2 that partition overruns and theta(0.5) stops mirroring
        // theta(0). Check the closure holds across the clamp boundary.
        for d in [0.50, 0.55, 0.60, 0.70] {
            let c = BoundTrimConfig { duty_factor: d, ..go2_cfg(0.7) };
            let a = c.sample(0.0).pitch;
            let b = c.sample(0.5).pitch;
            assert!((a + b).abs() < 1e-6,
                    "duty {d}: theta(0.5) must mirror theta(0), got {a} and {b}");
            assert!(c.sample(0.25).pitch.is_finite(), "duty {d}: mid-stance must be finite");
        }
    }

    #[test]
    fn f_z_total_grows_as_duty_factor_shrinks_below_half() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        for (duty, expected_fz) in [
            (0.50, 153.09),
            (0.45, 170.11),
            (0.40, 191.37),
            (0.35, 218.71),
            (0.30, 255.16),
            (0.25, 306.19),
        ] {
            cfg.duty_factor = duty;
            assert!(
                (cfg.f_z_total() - expected_fz).abs() < 0.1,
                "duty={duty}: f_z_total={:.2} expected {expected_fz:.2}",
                cfg.f_z_total()
            );
        }
    }

    /// `theta_boundary`'s closed-form `(θ(0), θ̇(0))` matches the
    /// Python reference at each swept `duty_factor`, including the
    /// `duty=0.5` case reproducing `θ(0)=0` exactly (no flight, no
    /// change from the original single-parabola derivation).
    #[test]
    fn theta_boundary_matches_flight_phase_closed_form() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        for (duty, expected_th0, expected_thd0) in [
            (0.50, 0.0, 0.40170),
            (0.45, 0.00181, 0.40170),
            (0.40, 0.00362, 0.40170),
            (0.35, 0.00542, 0.40170),
            (0.30, 0.00723, 0.40170),
            (0.25, 0.00904, 0.40170),
        ] {
            cfg.duty_factor = duty;
            let f_x = cfg.f_x_clipped();
            let (th0, thd0) = cfg.theta_boundary(f_x);
            assert!(
                (th0 - expected_th0).abs() < 0.001,
                "duty={duty}: th0={th0:.5} expected {expected_th0:.5}"
            );
            assert!(
                (thd0 - expected_thd0).abs() < 0.001,
                "duty={duty}: thd0={thd0:.5} expected {expected_thd0:.5}"
            );
        }
    }

    /// `theta_peak` (the 3-candidate generalization) matches the
    /// Python reference across the same duty sweep.
    #[test]
    fn theta_peak_matches_flight_phase_closed_form() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        for (duty, expected_peak) in [
            (0.50, 0.00904),
            (0.45, 0.00994),
            (0.40, 0.01085),
            (0.35, 0.01175),
            (0.30, 0.01265),
            (0.25, 0.01356),
        ] {
            cfg.duty_factor = duty;
            let f_x = cfg.f_x_clipped();
            let peak = cfg.theta_peak(f_x);
            assert!(
                (peak - expected_peak).abs() < 0.001,
                "duty={duty}: theta_peak={peak:.5} expected {expected_peak:.5}"
            );
        }
    }

    /// During the flight segment of a `duty_factor<0.5` half-cycle,
    /// `sample()` must report zero force on both axes -- no feet on
    /// the ground to push against.
    #[test]
    fn flight_phase_sample_has_zero_force() {
        let mut cfg = go2_cfg(0.7);
        cfg.cycle_period_s = 0.18;
        cfg.duty_factor = 0.4; // T_st=0.072, T_flight=0.018, half-cycle=0.09
        // Front-pair flight: local time s in (T_st, T/2) -> cycle_phase in (0.4, 0.5).
        let sample = cfg.sample(0.45);
        assert_eq!(sample.f_x_per_leg, 0.0);
        assert_eq!(sample.f_z_per_leg, 0.0);
        // Rear-pair flight: cycle_phase in (0.9, 1.0).
        let sample = cfg.sample(0.95);
        assert_eq!(sample.f_x_per_leg, 0.0);
        assert_eq!(sample.f_z_per_leg, 0.0);
    }

    /// At `duty_factor=0.5` (no flight phase, `T_flight=0`), `sample`'s
    /// generalized stance/flight branch must reproduce the original
    /// single-parabola trajectory exactly at several phases -- this is
    /// the key regression check that the `s` mapping fix (`local_frac
    /// * cycle_period_s` instead of the old `local_frac * 2 * t_st()`,
    /// which only coincided with the correct formula at duty=0.5)
    /// didn't change duty=0.5 behaviour.
    #[test]
    fn duty_one_half_sample_matches_original_single_parabola() {
        let cfg = go2_cfg(0.7);
        let f_x = cfg.f_x_clipped();
        let alpha_p = -(cfg.h0 * f_x + cfg.r_x * cfg.f_z_total()) / cfg.inertia_yy;
        let t_st = cfg.cycle_period_s * 0.5;
        for cycle_phase in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] {
            let local_frac = if cycle_phase < 0.5 { cycle_phase } else { cycle_phase - 0.5 };
            let s = local_frac * 2.0 * t_st; // old formula, exact at duty=0.5
            let expected_theta = (alpha_p / 2.0) * s * (s - t_st);
            let expected_theta_dot = alpha_p * (s - t_st / 2.0);
            let sample = cfg.sample(cycle_phase);
            let (theta, theta_dot) = if cycle_phase < 0.5 {
                (sample.pitch / cfg.sign, sample.pitch_rate / cfg.sign)
            } else {
                (-sample.pitch / cfg.sign, -sample.pitch_rate / cfg.sign)
            };
            assert!(
                (theta - expected_theta).abs() < 1e-9,
                "cycle_phase={cycle_phase}: theta={theta:.6} expected {expected_theta:.6}"
            );
            assert!(
                (theta_dot - expected_theta_dot).abs() < 1e-9,
                "cycle_phase={cycle_phase}: theta_dot={theta_dot:.6} expected {expected_theta_dot:.6}"
            );
        }
    }
}
