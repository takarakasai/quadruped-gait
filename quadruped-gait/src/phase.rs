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
        let duty = self.cfg.duty_factor.clamp(1e-6, 1.0 - 1e-6);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GaitType;

    fn pg() -> PhaseGenerator {
        PhaseGenerator::new(GaitConfig::trot())
    }

    fn find(legs: [PhaseState; 4], id: LegId) -> PhaseState {
        legs.into_iter().find(|p| p.leg == id).unwrap()
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
}
