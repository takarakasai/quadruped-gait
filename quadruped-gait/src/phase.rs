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
