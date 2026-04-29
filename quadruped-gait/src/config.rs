//! Configuration types: leg kinematics, gait parameters, leg identifiers.

use nalgebra::Vector3;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The four legs of a quadruped. Used as both a fixed-arity index into
/// per-leg arrays and as a key in serialised configs.
///
/// Naming convention follows CHAMP / most quadruped conventions:
/// `FL` = Front-Left, `FR` = Front-Right, `RL` = Rear-Left, `RR` = Rear-Right.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LegId {
    FL,
    FR,
    RL,
    RR,
}

impl LegId {
    pub const ALL: [LegId; 4] = [LegId::FL, LegId::FR, LegId::RL, LegId::RR];

    pub fn label(self) -> &'static str {
        match self {
            LegId::FL => "FL",
            LegId::FR => "FR",
            LegId::RL => "RL",
            LegId::RR => "RR",
        }
    }

    /// Whether the leg is on the front (vs rear) half of the body. Used by
    /// the trot phase generator to assign anti-phase pairs (FL+RR vs FR+RL).
    pub fn is_front(self) -> bool {
        matches!(self, LegId::FL | LegId::FR)
    }

    /// Whether the leg is on the left (vs right) side. Combined with
    /// [`Self::is_front`] forms the body-quadrant identifier.
    pub fn is_left(self) -> bool {
        matches!(self, LegId::FL | LegId::RL)
    }
}

/// Body-frame velocity command driving the gait. `vx` is forward, `vy` is
/// lateral (to the body's left), `wz` is yaw rate (counter-clockwise viewed
/// from above). All in SI.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VelocityCmd {
    /// Forward velocity (m/s).
    pub vx: f64,
    /// Lateral velocity, body-left positive (m/s).
    pub vy: f64,
    /// Yaw rate (rad/s).
    pub wz: f64,
}

impl VelocityCmd {
    pub const fn zero() -> Self {
        Self { vx: 0.0, vy: 0.0, wz: 0.0 }
    }

    /// True if the command is exactly zero. Used by the controller to gate
    /// the "stand still — feet on the ground, no swing" mode.
    pub fn is_zero(&self) -> bool {
        self.vx == 0.0 && self.vy == 0.0 && self.wz == 0.0
    }

    /// L2 magnitude of the linear part. Used by the footstep planner to
    /// decide step length scaling.
    pub fn linear_speed(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }
}

/// High-level gait family. Only [`GaitType::Trot`] is implemented in v0.1;
/// other variants are kept as enum members so configs remain forward-
/// compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GaitType {
    /// Diagonal pairs in phase: FL+RR ↔ FR+RL. Stable, agile, common.
    Trot,
    /// All four legs phased ¼ cycle apart. Slowest, most statically stable.
    Walk,
    /// Lateral pairs in phase: FL+RL ↔ FR+RR. Camel-style.
    Pace,
    /// Front pair + rear pair anti-phase. Bunny hop.
    Bound,
}

impl GaitType {
    pub const ALL: [GaitType; 4] = [
        GaitType::Trot,
        GaitType::Walk,
        GaitType::Pace,
        GaitType::Bound,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GaitType::Trot => "Trot",
            GaitType::Walk => "Walk",
            GaitType::Pace => "Pace",
            GaitType::Bound => "Bound",
        }
    }

    /// Phase offset (cycles) for each leg. Cycle is normalised to [0, 1).
    /// `0` means "in phase with the cycle start", `0.5` means "half-cycle
    /// later". The PhaseGenerator adds this to the global cycle phase to
    /// derive each leg's per-cycle position.
    pub fn phase_offsets(self) -> [(LegId, f64); 4] {
        match self {
            GaitType::Trot => [
                (LegId::FL, 0.0),
                (LegId::RR, 0.0),
                (LegId::FR, 0.5),
                (LegId::RL, 0.5),
            ],
            GaitType::Walk => [
                (LegId::FL, 0.0),
                (LegId::RR, 0.25),
                (LegId::FR, 0.5),
                (LegId::RL, 0.75),
            ],
            GaitType::Pace => [
                (LegId::FL, 0.0),
                (LegId::RL, 0.0),
                (LegId::FR, 0.5),
                (LegId::RR, 0.5),
            ],
            GaitType::Bound => [
                (LegId::FL, 0.0),
                (LegId::FR, 0.0),
                (LegId::RL, 0.5),
                (LegId::RR, 0.5),
            ],
        }
    }

    /// Default duty factor (fraction of cycle each foot spends in stance).
    /// 0.5 = symmetric (trot/pace/bound); higher = more stable (walk).
    pub fn default_duty_factor(self) -> f64 {
        match self {
            GaitType::Trot => 0.5,
            GaitType::Walk => 0.75,
            GaitType::Pace => 0.5,
            GaitType::Bound => 0.5,
        }
    }
}

/// Top-level gait configuration. Independent of the robot's kinematics
/// (those live in [`KinematicsConfig`]); this struct only describes
/// timing and footstep sizing.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaitConfig {
    pub gait_type: GaitType,
    /// Total cycle period (s). One full leg phase from stance-start back to
    /// stance-start.
    pub cycle_period_s: f64,
    /// Fraction of [0, 1) the foot spends on the ground. 0.5 ↔ trot at
    /// rest, increases for slower walks. Per-leg phase function uses this
    /// to discriminate stance vs swing.
    pub duty_factor: f64,
    /// Peak swing-foot height above the nominal stance plane (m).
    pub swing_height_m: f64,
    /// Maximum forward step length (m). Footstep planner clamps to this so
    /// the robot can't ask for a larger swing than its leg geometry allows.
    pub max_step_length_m: f64,
}

impl GaitConfig {
    /// Sensible default for a small quadruped like Mini Pupper / Solo.
    pub fn trot() -> Self {
        Self {
            gait_type: GaitType::Trot,
            cycle_period_s: 0.4,
            duty_factor: 0.5,
            swing_height_m: 0.04,
            max_step_length_m: 0.10,
        }
    }

    pub fn with_cycle_period(mut self, s: f64) -> Self {
        self.cycle_period_s = s.max(0.05);
        self
    }
    pub fn with_swing_height(mut self, m: f64) -> Self {
        self.swing_height_m = m.max(0.0);
        self
    }
    pub fn with_duty_factor(mut self, d: f64) -> Self {
        self.duty_factor = d.clamp(0.05, 0.95);
        self
    }
    pub fn with_max_step_length(mut self, m: f64) -> Self {
        self.max_step_length_m = m.max(0.0);
        self
    }
}

/// Per-leg geometric configuration. Determined by the user-provided foot
/// link plus auto-detection from the [`misarta`] model in articara.
///
/// The leg is assumed to be a serial 3-DOF Roll-Pitch-Pitch chain:
/// `body → hip_joint (Roll/X) → thigh_joint (Pitch/Y) → calf_joint (Pitch/Y) → foot`.
///
/// Coordinate convention (CHAMP-compatible):
/// - body frame: x forward, y left, z up
/// - hip frame: same orientation as body, translated by `hip_offset`
/// - In nominal stance with q = (0, 0, 0):
///   - thigh points straight down (-z)
///   - calf points straight down (-z), giving a fully-extended leg
/// - Positive thigh angle rotates the leg backward (knee forward)
/// - Positive calf angle bends the knee further (more flexed)
///
/// Joint sign conventions on real robots vary; the host application is
/// responsible for adapting URDF axis directions to this convention.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LegKinematics {
    pub leg: LegId,
    pub hip_joint: String,
    pub thigh_joint: String,
    pub calf_joint: String,
    pub foot_link: String,
    /// Translation from body origin to the hip-roll axis, in body frame.
    pub hip_offset: Vector3<f64>,
    /// Lateral offset from the hip-roll axis to the thigh-pitch axis,
    /// along body Y (positive = away from centerline). Often a few cm.
    pub hip_to_thigh_y: f64,
    /// Length of the upper segment (hip pitch axis → knee pitch axis), m.
    pub upper_leg_m: f64,
    /// Length of the lower segment (knee pitch axis → foot), m.
    pub lower_leg_m: f64,
    /// Nominal foot position in body frame at stance neutral. Used as the
    /// anchor that the footstep planner perturbs by Raibert offsets. By
    /// default equals `hip_offset + (0, ±hip_to_thigh_y, -(upper+lower))`
    /// (legs straight down) but the user may override (e.g. for a
    /// crouched standing height).
    pub nominal_foot_body: Vector3<f64>,
}

impl LegKinematics {
    /// Construct with explicit values. For auto-detection from a URDF, see
    /// the articara `gait::auto_detect_kinematics` helper.
    pub fn new(
        leg: LegId,
        hip_joint: String,
        thigh_joint: String,
        calf_joint: String,
        foot_link: String,
        hip_offset: Vector3<f64>,
        hip_to_thigh_y: f64,
        upper_leg_m: f64,
        lower_leg_m: f64,
    ) -> Self {
        // Default nominal foot: directly below the thigh pitch axis with
        // legs fully extended. Sign of the lateral component flips based
        // on which side of the body the leg is on.
        let lateral_sign = if leg.is_left() { 1.0 } else { -1.0 };
        let nominal = Vector3::new(
            hip_offset.x,
            hip_offset.y + lateral_sign * hip_to_thigh_y,
            hip_offset.z - (upper_leg_m + lower_leg_m),
        );
        Self {
            leg,
            hip_joint,
            thigh_joint,
            calf_joint,
            foot_link,
            hip_offset,
            hip_to_thigh_y,
            upper_leg_m,
            lower_leg_m,
            nominal_foot_body: nominal,
        }
    }
}

/// Complete kinematics description: one [`LegKinematics`] per leg. Stored
/// as four explicit fields rather than a HashMap so consumers can rely on
/// presence at compile time and the order is canonical (FL/FR/RL/RR).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KinematicsConfig {
    pub fl: LegKinematics,
    pub fr: LegKinematics,
    pub rl: LegKinematics,
    pub rr: LegKinematics,
}

impl KinematicsConfig {
    /// Convenience getter dispatching by [`LegId`].
    pub fn leg(&self, id: LegId) -> &LegKinematics {
        match id {
            LegId::FL => &self.fl,
            LegId::FR => &self.fr,
            LegId::RL => &self.rl,
            LegId::RR => &self.rr,
        }
    }

    pub fn legs(&self) -> [&LegKinematics; 4] {
        [&self.fl, &self.fr, &self.rl, &self.rr]
    }
}

/// Default leg-foot link names assumed when the user hasn't customised
/// them in the setup UI. Match the most common convention used across
/// open-source quadruped URDFs (Solo, Mini Pupper, ETH ANYmal exports, …).
pub const DEFAULT_FOOT_LINKS: [(LegId, &str); 4] = [
    (LegId::FL, "FL_foot"),
    (LegId::FR, "FR_foot"),
    (LegId::RL, "RL_foot"),
    (LegId::RR, "RR_foot"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trot_phase_pairs() {
        let offsets = GaitType::Trot.phase_offsets();
        // FL + RR are diagonal pair A; FR + RL are diagonal pair B.
        let mut by_leg = std::collections::HashMap::new();
        for (leg, off) in offsets {
            by_leg.insert(leg, off);
        }
        assert_eq!(by_leg[&LegId::FL], by_leg[&LegId::RR]);
        assert_eq!(by_leg[&LegId::FR], by_leg[&LegId::RL]);
        assert_ne!(by_leg[&LegId::FL], by_leg[&LegId::FR]);
    }

    #[test]
    fn nominal_foot_below_hip() {
        let kin = LegKinematics::new(
            LegId::FL,
            "FL_hip".into(),
            "FL_thigh".into(),
            "FL_calf".into(),
            "FL_foot".into(),
            Vector3::new(0.18, 0.05, 0.0),
            0.04,
            0.18,
            0.18,
        );
        // Default: x stays, y shifts outward (left for FL), z goes down by
        // upper + lower leg lengths.
        assert!((kin.nominal_foot_body.x - 0.18).abs() < 1e-9);
        assert!((kin.nominal_foot_body.y - 0.09).abs() < 1e-9);
        assert!((kin.nominal_foot_body.z - (-0.36)).abs() < 1e-9);
    }
}
