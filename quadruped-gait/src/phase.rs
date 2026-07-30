//! Phase generator for periodic gaits.
//!
//! Maintains a single global cycle counter `cycle_phase ∈ [0, 1)` that
//! advances by `dt / cycle_period` each tick. Each leg derives its own
//! per-cycle phase by adding the gait's leg-specific offset (see
//! [`crate::config::GaitType::phase_offsets`]).
//!
//! Each per-leg phase is then split into stance (when on the ground) and
//! swing (when in the air) by the duty factor:
//!
//! ```text
//! per_leg_phase = (cycle_phase + offset) mod 1.0
//! is_stance     = per_leg_phase < duty_factor
//! stance_frac   = per_leg_phase / duty_factor                 (0..1 in stance)
//! swing_frac    = (per_leg_phase - duty_factor) /
//!                 (1 - duty_factor)                            (0..1 in swing)
//! ```

use crate::config::{GaitConfig, LegId, VelocityCmd};

/// Per-leg phase decomposition. Either Stance or Swing, with a normalised
/// fraction in [0, 1] indicating progress within that sub-phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseState {
    pub leg: LegId,
    /// Current normalised position in the leg's full cycle, [0, 1).
    pub cycle_position: f64,
    /// True if the leg is currently in the stance (on-ground) sub-phase.
    pub is_stance: bool,
    /// Progress through the current sub-phase, [0, 1].
    pub sub_fraction: f64,
}

/// Stateful phase generator. One per gait controller.
#[derive(Clone, Debug)]
pub struct PhaseGenerator {
    /// Current global cycle phase, [0, 1).
    cycle_phase: f64,
    cfg: GaitConfig,
    /// When the last velocity command was zero we hold the cycle phase
    /// frozen at the start of stance for every leg, otherwise the
    /// controller would generate phantom swing motions while standing.
    /// Tracked across ticks so a brief stop doesn't reset the phase.
    holding: bool,
}

impl PhaseGenerator {
    pub fn new(cfg: GaitConfig) -> Self {
        Self {
            cycle_phase: 0.0,
            cfg,
            holding: true,
        }
    }

    pub fn config(&self) -> &GaitConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: GaitConfig) {
        self.cfg = cfg;
    }

    /// Force the generator back to the cycle origin (cycle_phase = 0).
    /// Useful when the user explicitly stops the gait so the next start
    /// begins from a deterministic state.
    pub fn reset(&mut self) {
        self.cycle_phase = 0.0;
        self.holding = true;
    }

    /// Read-only access to the current cycle phase.
    pub fn cycle_phase(&self) -> f64 {
        self.cycle_phase
    }

    /// Advance the global cycle phase by `dt`. Stops advancing when the
    /// velocity command is zero so the legs settle in stance instead of
    /// continuing to swing in place.
    pub fn advance(&mut self, dt: f64, vel: &VelocityCmd) {
        if vel.is_zero() {
            self.holding = true;
            return;
        }
        self.holding = false;
        let period = self.cfg.cycle_period_s.max(1e-6);
        self.cycle_phase = (self.cycle_phase + dt / period).rem_euclid(1.0);
    }

    /// Compute the per-leg [`PhaseState`] for every leg given the current
    /// global cycle phase. When holding (zero velocity), every leg is
    /// reported as fully in stance with `sub_fraction = 0` so downstream
    /// trajectory generators emit the static-stance pose.
    pub fn legs(&self) -> [PhaseState; 4] {
        let offsets = self.cfg.gait_type.phase_offsets();
        let mut out = [PhaseState {
            leg: LegId::FL,
            cycle_position: 0.0,
            is_stance: true,
            sub_fraction: 0.0,
        }; 4];

        for (i, (leg, offset)) in offsets.into_iter().enumerate() {
            let pos = if self.holding {
                0.0
            } else {
                (self.cycle_phase + offset).rem_euclid(1.0)
            };
            // Per-leg duty: the rear pair may run a different one
            // (`duty_factor_rear_scale`), so this must be resolved inside
            // the loop rather than hoisted. Keyed on `leg`, not the loop
            // index, because `phase_offsets()`'s ordering is per gait type.
            let duty = self.cfg.duty_for(leg);
            let (is_stance, sub) = if self.holding {
                (true, 0.0)
            } else if pos < duty {
                (true, pos / duty)
            } else {
                (false, (pos - duty) / (1.0 - duty))
            };
            out[i] = PhaseState {
                leg,
                cycle_position: pos,
                is_stance,
                sub_fraction: sub,
            };
        }
        out
    }
}

/// Phase generator with **contact-driven correction**.
///
/// Wraps a [`PhaseGenerator`] (the nominal open-loop schedule) and
/// overrides each leg's `is_stance` flag based on the measured ground
/// reaction force per foot:
///
/// - **Early touchdown**: nominal schedule says swing, but the foot
///   is already loaded above `early_contact_threshold_n`. Switch the
///   leg to stance for the rest of the nominal swing window.
/// - **Late liftoff**: nominal says stance, but the foot has gone
///   unloaded below `late_liftoff_threshold_n`. Switch to swing.
///
/// Mirrors the `mode`-driven contact_flag handoff in
/// `legged_control` (where the OCS2 NMPC's `mode` index encodes the
/// active contact pattern), but here we rebuild it from real
/// physics measurements.
///
/// Without this layer, an open-loop trot will drift its phase clock
/// off the actual physics within ~1 cycle and the WBC's
/// `no_contact_motion` task will request `J·q̈ + J̇·v = 0` for a
/// foot that's actually swinging (or vice-versa) — an infeasibility
/// that the QP soaks up as constraint violation, kicking the body
/// into instability.
#[derive(Clone, Debug)]
pub struct ContactDrivenPhase {
    nominal: PhaseGenerator,
    /// Force threshold (N, world-z) above which an unscheduled
    /// contact counts as "early touchdown". Scale to robot weight:
    /// 0.10 · m·g is a reasonable default (a foot well-planted on
    /// the ground sees ≥ 0.25 · m·g during stance).
    pub early_contact_threshold_n: f64,
    /// Force threshold below which a scheduled stance leg counts as
    /// "late liftoff" (i.e. has already left the ground). Smaller
    /// than `early_contact_threshold_n` — slip / micro-bounce often
    /// drops the load briefly even when the foot stays planted.
    pub late_liftoff_threshold_n: f64,
}

impl ContactDrivenPhase {
    pub fn new(cfg: GaitConfig) -> Self {
        Self {
            nominal: PhaseGenerator::new(cfg),
            // Defaults assume a small (1–10 kg) quadruped where m·g ≈
            // 10–100 N. Hosts with heavier robots should bump these
            // thresholds proportionally; they're public for that
            // reason.
            early_contact_threshold_n: 5.0,
            late_liftoff_threshold_n: 1.0,
        }
    }

    pub fn config(&self) -> &GaitConfig {
        self.nominal.config()
    }

    pub fn set_config(&mut self, cfg: GaitConfig) {
        self.nominal.set_config(cfg);
    }

    pub fn reset(&mut self) {
        self.nominal.reset();
    }

    pub fn cycle_phase(&self) -> f64 {
        self.nominal.cycle_phase()
    }

    /// Advance the underlying nominal phase by `dt`. The `legs()`
    /// reading is *not yet* corrected for contact — call
    /// [`Self::corrected_legs`] right after with the per-foot ground
    /// reaction force to get the override-applied phases.
    pub fn advance(&mut self, dt: f64, vel: &VelocityCmd) {
        self.nominal.advance(dt, vel);
    }

    /// Per-leg phase **after** applying contact-driven `is_stance`
    /// overrides, given the per-foot world-z ground reaction force
    /// (`contact_force_z[slot]` ≥ 0, in N).
    ///
    /// `cycle_position` and `sub_fraction` are kept identical to the
    /// nominal generator's reading — the override is solely on
    /// `is_stance`. This keeps the swing-trajectory generators
    /// downstream working off the same time axis (so swing height
    /// curves don't snap discontinuously), while the WBC + MPC see
    /// the real contact pattern.
    pub fn corrected_legs(&self, contact_force_z: [f64; 4]) -> [PhaseState; 4] {
        Self::apply_correction(
            &self.nominal.legs(),
            contact_force_z,
            self.early_contact_threshold_n,
            self.late_liftoff_threshold_n,
        )
    }

    /// Stateless variant of [`Self::corrected_legs`]. Takes a nominal
    /// `[PhaseState; 4]` from any source (a `GaitController`'s `tick`
    /// output, an externally-driven schedule, etc.) and applies the
    /// same `is_stance` override rules.
    ///
    /// Useful when the caller already has nominal phases (so they
    /// don't want to maintain a parallel `ContactDrivenPhase` instance
    /// just for the correction logic).
    pub fn apply_correction(
        nominal: &[PhaseState; 4],
        contact_force_z: [f64; 4],
        early_contact_threshold_n: f64,
        late_liftoff_threshold_n: f64,
    ) -> [PhaseState; 4] {
        let mut legs = *nominal;
        for slot in 0..4 {
            let f = contact_force_z[slot];
            let nominal_stance = legs[slot].is_stance;
            if !nominal_stance && f > early_contact_threshold_n {
                // Early touchdown: foot landed before scheduled.
                legs[slot].is_stance = true;
            } else if nominal_stance && f < late_liftoff_threshold_n {
                // Late liftoff: foot already left ground.
                // Only override at non-zero sub_fraction so we don't
                // mistake the very-first stance tick (foot mid-air,
                // about to land) for a liftoff.
                if legs[slot].sub_fraction > 0.05 {
                    legs[slot].is_stance = false;
                }
            }
        }
        legs
    }

    /// Pass-through to the nominal generator's [`PhaseGenerator::legs`]
    /// (no contact correction applied). Useful for diagnostic plots.
    pub fn nominal_legs(&self) -> [PhaseState; 4] {
        self.nominal.legs()
    }
}

/// Estimates a signed phase-timing error (seconds) per leg, from the
/// same measured GRF [`ContactDrivenPhase::apply_correction`] already
/// uses -- but symmetric and quantitative where that function is
/// direction-limited and boolean-only.
///
/// `apply_correction` only ever detects the real gait running FASTER
/// than the nominal clock (early touchdown, early liftoff-from-
/// stance) -- there's no detection of the opposite (the real gait
/// running SLOWER: a touchdown that hasn't happened yet even though
/// the nominal schedule already switched to stance, or a foot that
/// stays loaded past the nominal stance-to-swing switch). This
/// tracker adds that missing direction, at the cost of a small amount
/// of cross-tick state (`apply_correction` is stateless) -- per leg,
/// whether the nominal stance/swing window flipped since last tick,
/// and whether a "too slow" event is pending resolution once flipped
/// (its magnitude isn't known at the transition instant -- only once
/// the foot finally does what the nominal schedule already expected).
///
/// Motivation: `articara/ref/wbc_comparison.md` Sec.5bk found the
/// (boolean, one-directional) `apply_correction` mismatch rate
/// correlates strongly with Bound's sparse cmd_vx instability points
/// (Sec.5bi) -- the natural next step is a slow feedback loop that
/// nudges `cycle_period_s` toward whatever period the real contact
/// timing implies, which needs a signed, quantitative error signal
/// symmetric in both directions. See Sec.5bl for the calibration.
#[derive(Clone, Debug)]
pub struct PhaseErrorTracker {
    prev_nominal_stance: [bool; 4],
    /// Set at a stance→swing transition if the foot was still loaded
    /// at that instant (stance overran into nominal swing); cleared
    /// once the foot finally unloads and the (positive, lengthen)
    /// error fires.
    stance_overrun_pending: [bool; 4],
    /// Set at a swing→stance transition if the foot was NOT yet
    /// loaded at that instant (touchdown hasn't happened yet even
    /// though nominal already switched to stance); cleared once the
    /// foot finally loads and the (positive, lengthen) error fires.
    late_touchdown_pending: [bool; 4],
    /// Whether the "too fast" event (early touchdown / early liftoff)
    /// has already fired for the current window -- keeps those two
    /// cases edge-triggered (fire once, at first detection) to match
    /// the "too slow" cases' one-shot nature, instead of re-firing
    /// (with a shrinking magnitude) every tick the condition holds.
    fast_event_fired: [bool; 4],
}

impl Default for PhaseErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseErrorTracker {
    pub fn new() -> Self {
        Self {
            prev_nominal_stance: [true; 4],
            stance_overrun_pending: [false; 4],
            late_touchdown_pending: [false; 4],
            fast_event_fired: [false; 4],
        }
    }

    /// Call once per tick, in lockstep with the same `nominal` phases
    /// and `contact_force_z` (world-z GRF, N) [`ContactDrivenPhase::
    /// apply_correction`] would receive. Returns, per leg, `Some(err)`
    /// on the tick a mismatch event is first detected (`None`
    /// otherwise) -- `err` is a signed seconds estimate: negative
    /// means the real gait ran that much FASTER than `cycle_period_s`
    /// assumes over the current sub-phase (shorten the period),
    /// positive means SLOWER (lengthen it).
    ///
    /// Four mutually-exclusive events, one per combination of (which
    /// sub-phase nominal is in) x (which direction the mismatch runs):
    /// early touchdown / early liftoff (both "too fast", already
    /// covered non-quantitatively by [`ContactDrivenPhase::
    /// apply_correction`]), late touchdown / stance-overrun-into-swing
    /// (both "too slow", new). The two "too slow" cases are detected
    /// as *pending* at the transition instant (since at that instant
    /// we don't yet know how much longer reality will take) and
    /// resolved -- with the actual seconds error -- once the foot
    /// finally does what the nominal schedule already expected.
    pub fn observe(
        &mut self,
        nominal: &[PhaseState; 4],
        contact_force_z: [f64; 4],
        early_contact_threshold_n: f64,
        late_liftoff_threshold_n: f64,
        cycle_period_s: f64,
        duty_factors: [f64; 4],
    ) -> [Option<f64>; 4] {
        let mut out = [None; 4];
        for slot in 0..4 {
            // Per-slot, since front and rear may run different duties
            // (`GaitConfig::duty_factor_rear_scale`). Passing one scalar here
            // would size the rear pair's stance/swing windows wrongly and so
            // misreport exactly the lateness this tracker exists to measure.
            let duty_factor = duty_factors[slot];
            let ps = nominal[slot];
            let f = contact_force_z[slot];
            let loaded = f > early_contact_threshold_n;
            let unloaded = f < late_liftoff_threshold_n;

            if ps.is_stance != self.prev_nominal_stance[slot] {
                if ps.is_stance {
                    // swing -> stance: if not loaded THIS tick, touchdown
                    // is running late relative to the clock.
                    self.late_touchdown_pending[slot] = !loaded;
                } else {
                    // stance -> swing: if still loaded THIS tick, stance
                    // ran over into what the clock already calls swing.
                    self.stance_overrun_pending[slot] = loaded;
                }
                self.fast_event_fired[slot] = false;
            }

            if self.late_touchdown_pending[slot] {
                if loaded {
                    out[slot] = Some(ps.sub_fraction * duty_factor * cycle_period_s);
                    self.late_touchdown_pending[slot] = false;
                }
            } else if self.stance_overrun_pending[slot] {
                if unloaded {
                    out[slot] = Some(ps.sub_fraction * (1.0 - duty_factor) * cycle_period_s);
                    self.stance_overrun_pending[slot] = false;
                }
            } else if ps.is_stance {
                if unloaded && ps.sub_fraction > 0.05 && !self.fast_event_fired[slot] {
                    // Early liftoff (was on-time or late at touchdown,
                    // per the branches above, and is now unloading
                    // before the nominal stance window ends).
                    out[slot] = Some(-(1.0 - ps.sub_fraction) * duty_factor * cycle_period_s);
                    self.fast_event_fired[slot] = true;
                }
            } else if loaded && !self.fast_event_fired[slot] {
                // Early touchdown (was on-time at the swing handoff,
                // per the branches above, and is now loaded before the
                // nominal swing window ends).
                out[slot] = Some(-(1.0 - ps.sub_fraction) * (1.0 - duty_factor) * cycle_period_s);
                self.fast_event_fired[slot] = true;
            }

            self.prev_nominal_stance[slot] = ps.is_stance;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pg() -> PhaseGenerator {
        PhaseGenerator::new(GaitConfig::trot())
    }

    fn find(legs: [PhaseState; 4], id: LegId) -> PhaseState {
        legs.into_iter().find(|p| p.leg == id).unwrap()
    }

    /// `duty_factor_rear_scale` must put the two pairs on different clocks,
    /// and it must be a no-op at 1.0 -- both halves matter, since every
    /// existing caller relies on the second.
    #[test]
    fn rear_duty_scale_splits_the_schedule_and_is_a_no_op_at_one() {
        let bound = |scale: f64| {
            let mut cfg = GaitConfig::for_type(crate::config::GaitType::Bound);
            cfg.duty_factor = 0.5;
            cfg.duty_factor_rear_scale = scale;
            cfg
        };

        // At 1.0 the pairs tile: front stance over [0, 0.5), rear over
        // [0.5, 1.0), and no instant is either doubly supported or empty.
        let sym = bound(1.0);
        for i in 0..100 {
            let phase = i as f64 / 100.0;
            let mut g = PhaseGenerator::new(sym.clone());
            g.advance(phase * sym.cycle_period_s, &VelocityCmd { vx: 0.5, vy: 0.0, wz: 0.0 });
            let legs = g.legs();
            let front = find(legs, LegId::FL).is_stance;
            let rear = find(legs, LegId::RL).is_stance;
            assert!(front != rear, "phase {phase}: pairs must alternate at scale 1.0");
        }

        // At 0.8 the rear's stance window shrinks to 0.4 of the cycle, so
        // [0.9, 1.0) is a genuine flight phase -- and the rear's SWING
        // window grows from 0.5*T to 0.6*T, which is the whole point.
        let asym = bound(0.8);
        assert_eq!(asym.duty_for(LegId::FL), 0.5);
        assert_eq!(asym.duty_for(LegId::RL), 0.4);
        assert_eq!(asym.duty_factors(), [0.5, 0.5, 0.4, 0.4]);

        let stance_at = |phase: f64| {
            let mut g = PhaseGenerator::new(asym.clone());
            g.advance(phase * asym.cycle_period_s, &VelocityCmd { vx: 0.5, vy: 0.0, wz: 0.0 });
            let legs = g.legs();
            (find(legs, LegId::FL).is_stance, find(legs, LegId::RL).is_stance)
        };
        assert_eq!(stance_at(0.25), (true, false), "front-only");
        assert_eq!(stance_at(0.70), (false, true), "rear-only");
        assert_eq!(stance_at(0.95), (false, false), "flight");
    }

    #[test]
    fn zero_command_holds_all_in_stance() {
        let mut g = pg();
        g.advance(0.5, &VelocityCmd::zero()); // half a period at zero
        for ps in g.legs() {
            assert!(ps.is_stance);
            assert_eq!(ps.sub_fraction, 0.0);
        }
    }

    #[test]
    fn trot_diagonal_pairs_in_phase() {
        let mut g = pg();
        let cmd = VelocityCmd { vx: 0.3, vy: 0.0, wz: 0.0 };
        // Advance by 1/8 of a cycle so legs are off-zero and sub-fraction
        // is mid-stance for one diagonal pair / mid-swing for the other.
        g.advance(g.cfg.cycle_period_s * 0.125, &cmd);
        let legs = g.legs();
        let fl = find(legs, LegId::FL);
        let rr = find(legs, LegId::RR);
        let fr = find(legs, LegId::FR);
        let rl = find(legs, LegId::RL);

        // Diagonal pair A (FL+RR) at phase 0.125 → stance
        assert!(fl.is_stance);
        assert!(rr.is_stance);
        assert_eq!(fl.cycle_position, rr.cycle_position);
        // Diagonal pair B (FR+RL) at phase 0.625 → swing (after duty 0.5)
        assert!(!fr.is_stance);
        assert!(!rl.is_stance);
        assert_eq!(fr.cycle_position, rl.cycle_position);
    }

    #[test]
    fn cycle_wraps_modulo_one() {
        let mut g = pg();
        let cmd = VelocityCmd { vx: 0.3, ..Default::default() };
        // Advance 2.7 cycles' worth of time at this period.
        let total = g.cfg.cycle_period_s * 2.7;
        g.advance(total, &cmd);
        let p = g.cycle_phase();
        assert!(p >= 0.0 && p < 1.0, "cycle phase wrapped to {p}");
        assert!((p - 0.7).abs() < 1e-9, "expected ≈0.7, got {p}");
    }

    /// When the per-foot forces are **consistent** with the nominal
    /// schedule (stance legs loaded, swing legs unloaded), the
    /// correction must be a no-op.
    #[test]
    fn contact_driven_passthrough_when_consistent() {
        let mut g = ContactDrivenPhase::new(GaitConfig::trot());
        let cmd = VelocityCmd { vx: 0.3, ..Default::default() };
        g.advance(g.config().cycle_period_s * 0.25, &cmd);
        let nominal = g.nominal_legs();
        let mut force = [0.0; 4];
        for i in 0..4 {
            if nominal[i].is_stance {
                force[i] = 50.0; // well-loaded stance
            }
        }
        let corrected = g.corrected_legs(force);
        for i in 0..4 {
            assert_eq!(nominal[i].is_stance, corrected[i].is_stance);
            assert_eq!(nominal[i].cycle_position, corrected[i].cycle_position);
        }
    }

    /// Early touchdown: a leg in nominal swing receives a force above
    /// the threshold → corrected reports stance.
    #[test]
    fn contact_driven_early_touchdown_flips_to_stance() {
        let mut g = ContactDrivenPhase::new(GaitConfig::trot());
        g.early_contact_threshold_n = 5.0;
        let cmd = VelocityCmd { vx: 0.3, ..Default::default() };
        // Advance to a mid-swing position for FR (offset 0.5, duty 0.5
        // → at cycle_phase 0.625, FR is at 0.125 → sub-cycle but still
        // in swing). Hard-set so the test doesn't depend on numerics:
        g.advance(g.config().cycle_period_s * 0.625, &cmd);
        let nominal = g.nominal_legs();
        // Find a swing leg.
        let swing_slot = (0..4).find(|&i| !nominal[i].is_stance).expect("a swing leg");
        let mut force = [0.0; 4];
        force[swing_slot] = 50.0; // well above threshold
        let corrected = g.corrected_legs(force);
        assert!(corrected[swing_slot].is_stance,
            "swing leg with f_z = 50 should be flipped to stance");
    }

    /// Late liftoff: a leg in nominal mid-stance whose force drops
    /// below the threshold → corrected reports swing.
    #[test]
    fn contact_driven_late_liftoff_flips_to_swing() {
        let mut g = ContactDrivenPhase::new(GaitConfig::trot());
        g.late_liftoff_threshold_n = 1.0;
        let cmd = VelocityCmd { vx: 0.3, ..Default::default() };
        // Mid-stance position for one of the diagonal pairs.
        g.advance(g.config().cycle_period_s * 0.25, &cmd);
        let nominal = g.nominal_legs();
        let stance_slot = (0..4)
            .find(|&i| nominal[i].is_stance && nominal[i].sub_fraction > 0.1)
            .expect("a mid-stance leg");
        let force = [0.0; 4]; // unloaded
        let corrected = g.corrected_legs(force);
        assert!(!corrected[stance_slot].is_stance,
            "mid-stance leg with f_z = 0 should be flipped to swing");
    }

    /// First-tick of stance (sub_fraction ≈ 0) must NOT be reported as
    /// liftoff just because the force is low — the foot is in the
    /// air about to land in that instant. Guards a subtle false
    /// positive that would otherwise oscillate at every stance entry.
    #[test]
    fn contact_driven_late_liftoff_ignores_stance_entry() {
        let g = ContactDrivenPhase::new(GaitConfig::trot());
        // Hold (vel = 0) → all legs are in stance with sub_fraction = 0.
        let force = [0.0; 4];
        let corrected = g.corrected_legs(force);
        for ps in corrected.iter() {
            assert!(ps.is_stance,
                "stance entry tick must stay stance even when unloaded");
        }
    }

    #[test]
    fn duty_split_at_boundary() {
        // With duty = 0.5, leg at position 0.5 should be at swing-start.
        // We can't set position directly so manually advance there.
        let mut g = pg();
        let cmd = VelocityCmd { vx: 0.3, ..Default::default() };
        g.advance(g.cfg.cycle_period_s * 0.5, &cmd);
        let fl = find(g.legs(), LegId::FL);
        // FL has offset 0 so cycle_position == cycle_phase ≈ 0.5; with
        // duty 0.5 this is the boundary. Floating-point may put it just
        // below or above so we accept either side.
        assert!((fl.cycle_position - 0.5).abs() < 1e-9);
    }

    // --- PhaseErrorTracker (Sec.5bl, local doc) --------------------------

    const T_ERR: f64 = 0.30;
    const DUTY_ERR: f64 = 0.5;
    const EARLY_N: f64 = 5.0;
    const LATE_N: f64 = 1.0;

    fn err_ps(is_stance: bool, sub_fraction: f64) -> PhaseState {
        PhaseState { leg: LegId::FL, cycle_position: 0.0, is_stance, sub_fraction }
    }

    fn observe1(
        tracker: &mut PhaseErrorTracker,
        is_stance: bool,
        sub_fraction: f64,
        force_n: f64,
    ) -> Option<f64> {
        let nominal = [err_ps(is_stance, sub_fraction); 4];
        let force = [force_n; 4];
        tracker.observe(&nominal, force, EARLY_N, LATE_N, T_ERR, [DUTY_ERR; 4])[0]
    }

    /// Force always agrees with the nominal schedule (loaded in
    /// stance, unloaded in swing) -- no event should ever fire.
    #[test]
    fn phase_error_tracker_silent_when_consistent() {
        let mut tracker = PhaseErrorTracker::new();
        for i in 0..=10 {
            let sub = i as f64 / 10.0;
            assert!(observe1(&mut tracker, true, sub, 50.0).is_none(), "stance sub={sub}");
        }
        for i in 0..=10 {
            let sub = i as f64 / 10.0;
            assert!(observe1(&mut tracker, false, sub, 0.0).is_none(), "swing sub={sub}");
        }
    }

    /// Foot touches down (loaded) partway through nominal swing --
    /// "too fast", negative error, magnitude from the remaining
    /// nominal swing time.
    #[test]
    fn phase_error_tracker_early_touchdown() {
        let mut tracker = PhaseErrorTracker::new();
        // Prime: consistent stance, then a clean (unloaded) swing entry.
        assert!(observe1(&mut tracker, true, 0.99, 50.0).is_none());
        assert!(observe1(&mut tracker, false, 0.0, 0.0).is_none());
        assert!(observe1(&mut tracker, false, 0.3, 0.0).is_none());
        // Touches down early, at swing sub_fraction 0.8 (should run to 1.0).
        let err = observe1(&mut tracker, false, 0.8, 50.0).expect("expected early-touchdown event");
        let expected = -(1.0 - 0.8) * (1.0 - DUTY_ERR) * T_ERR;
        assert!((err - expected).abs() < 1e-9, "err={err} expected={expected}");
        // Must not re-fire on a later tick in the same window.
        assert!(observe1(&mut tracker, false, 0.9, 50.0).is_none());
    }

    /// Foot unloads (lifts off) partway through nominal stance, having
    /// been loaded earlier in that same window -- "too fast", negative
    /// error.
    #[test]
    fn phase_error_tracker_early_liftoff() {
        let mut tracker = PhaseErrorTracker::new();
        assert!(observe1(&mut tracker, true, 0.0, 50.0).is_none());
        assert!(observe1(&mut tracker, true, 0.3, 50.0).is_none());
        let err = observe1(&mut tracker, true, 0.8, 0.0).expect("expected early-liftoff event");
        let expected = -(1.0 - 0.8) * DUTY_ERR * T_ERR;
        assert!((err - expected).abs() < 1e-9, "err={err} expected={expected}");
    }

    /// Swing→stance transition happens, but the foot hasn't touched
    /// down yet -- "too slow", positive error, resolved once it
    /// finally loads.
    #[test]
    fn phase_error_tracker_late_touchdown() {
        let mut tracker = PhaseErrorTracker::new();
        assert!(observe1(&mut tracker, false, 0.9, 0.0).is_none());
        // Transition tick: nominal says stance, still unloaded -> pending.
        assert!(observe1(&mut tracker, true, 0.0, 0.0).is_none());
        assert!(observe1(&mut tracker, true, 0.2, 0.0).is_none());
        // Finally loads at stance sub_fraction 0.4.
        let err = observe1(&mut tracker, true, 0.4, 50.0).expect("expected late-touchdown event");
        let expected = 0.4 * DUTY_ERR * T_ERR;
        assert!((err - expected).abs() < 1e-9, "err={err} expected={expected}");
    }

    /// Stance→swing transition happens, but the foot is still loaded
    /// -- "too slow", positive error, resolved once it finally unloads.
    #[test]
    fn phase_error_tracker_stance_overrun() {
        let mut tracker = PhaseErrorTracker::new();
        assert!(observe1(&mut tracker, true, 0.9, 50.0).is_none());
        // Transition tick: nominal says swing, still loaded -> pending.
        assert!(observe1(&mut tracker, false, 0.0, 50.0).is_none());
        assert!(observe1(&mut tracker, false, 0.2, 50.0).is_none());
        // Finally unloads at swing sub_fraction 0.5.
        let err = observe1(&mut tracker, false, 0.5, 0.0).expect("expected stance-overrun event");
        let expected = 0.5 * (1.0 - DUTY_ERR) * T_ERR;
        assert!((err - expected).abs() < 1e-9, "err={err} expected={expected}");
    }
}
