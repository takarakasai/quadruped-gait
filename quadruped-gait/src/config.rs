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
    /// All four legs phased ¼ cycle apart. Slow, statically stable.
    Walk,
    /// Lateral pairs in phase: FL+RL ↔ FR+RR. Camel-style.
    Pace,
    /// Front pair + rear pair anti-phase. Bunny hop.
    Bound,
    /// **Static crawl.** Same diagonal-sequence offsets as [`Self::Walk`]
    /// (lift order RL → FR → RR → FL) but with a higher default duty
    /// factor (`0.85`) so swings never overlap and the support polygon
    /// always covers three feet. Pair with a long [`GaitConfig::cycle_period_s`]
    /// and a small [`GaitConfig::swing_height_m`] — the trunk barely
    /// rolls/pitches and moves forward smoothly. Slowest of the family.
    Crawl,
}

impl GaitType {
    pub const ALL: [GaitType; 5] = [
        GaitType::Trot,
        GaitType::Walk,
        GaitType::Pace,
        GaitType::Bound,
        GaitType::Crawl,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GaitType::Trot => "Trot",
            GaitType::Walk => "Walk",
            GaitType::Pace => "Pace",
            GaitType::Bound => "Bound",
            GaitType::Crawl => "Crawl",
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
            // Crawl shares Walk's diagonal-sequence offsets — see GaitType::Crawl.
            GaitType::Crawl => [
                (LegId::FL, 0.0),
                (LegId::RR, 0.25),
                (LegId::FR, 0.5),
                (LegId::RL, 0.75),
            ],
        }
    }

    /// Default duty factor (fraction of cycle each foot spends in stance).
    /// 0.5 = symmetric (trot/pace/bound); higher = more stable (walk/crawl).
    pub fn default_duty_factor(self) -> f64 {
        match self {
            GaitType::Trot => 0.5,
            GaitType::Walk => 0.75,
            GaitType::Pace => 0.5,
            GaitType::Bound => 0.5,
            GaitType::Crawl => 0.85,
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
    /// Fraction of each leg's **stance phase** that is treated as a
    /// load / unload transition. Reads only by the FullCentroidal
    /// controller's legged_control-parity path (C1 experiment): at
    /// touchdown the leg's GRF reference ramps from 0 → full over the
    /// first `transition_fraction · stance_duration`, and ramps back
    /// down → 0 over the last `transition_fraction · stance_duration`.
    /// The stance-no-slip constraint stays active throughout so the
    /// foot is still pinned; this is a **soft (cost-side)** smoother
    /// that helps the MPC plan a less impulsive loading trajectory.
    ///
    /// Default `0.0` ⇒ no ramping (legacy step behaviour). Reasonable
    /// values are `0.05`–`0.15`.
    pub transition_fraction: f64,
    /// **C1-2 experiment**: when `true` AND `transition_fraction > 0`,
    /// the FullCentroidal controller also applies the
    /// `stance_weight_at` curve to the per-leg per-step **f_z upper
    /// bound** that the MPC's friction-cone block enforces as a hard
    /// constraint. This is the constraint-side counterpart of the
    /// cost-side `transition_fraction` ramp: at touchdown the leg's
    /// `max_normal_force` ramps from 0 → full, forcing the MPC to
    /// redistribute load to other stance legs instead of
    /// instantaneously spiking the newly-touched-down leg.
    ///
    /// Default `false` keeps the legacy global `max_normal_force` as
    /// the only upper bound — backward compatible.
    pub transition_enforce_constraint: bool,
    /// **A3 — friction cone soft + slack** (FullCentroidal MPC).
    ///
    /// When `true`, the FullCentroidal controller flips
    /// [`crate::full_centroidal_mpc::FullCentroidalMpcConfig::friction_cone_soft`]
    /// for every MPC tick — the pyramid friction inequalities switch
    /// to a slack-relaxed form (`|f_x| ≤ μ·f_z + s`, `s ≥ 0`) with a
    /// quadratic cost on each slack. Useful when the GRF demand at
    /// the pyramid corner regularly reaches √2 of the SOC cone
    /// (lateral 4–6 N push regime on namiashi, per
    /// `diag_friction_cone_utilization`), because the hard form
    /// either over-tracks (clarabel returns AlmostSolved) or falls
    /// back to the reference solution.
    ///
    /// Default `false` ⇒ legacy hard pyramid, unchanged behaviour.
    pub friction_cone_soft: bool,
    /// Quadratic penalty weight applied to each friction-cone slack.
    /// See [`crate::full_centroidal_mpc::FullCentroidalMpcConfig::friction_cone_slack_penalty`].
    /// Only used when [`Self::friction_cone_soft`] is `true`. Default
    /// `1000.0` is the same as the MPC default — calibrated against
    /// the default `r_diag[GRF] = 1e-3`.
    pub friction_cone_slack_penalty: f64,
    /// **B3 — MPC warm-start** (FullCentroidal MPC).
    ///
    /// When `true`, the FullCentroidal controller mirrors the flag
    /// onto its MPC config so each solve seeds its SQP loop from the
    /// previous tick's predicted trajectory (shifted by one step) as
    /// the iter-0 reference. Reduces effective iterations needed for
    /// convergence — see
    /// [`crate::full_centroidal_mpc::FullCentroidalMpcConfig::warm_start`].
    ///
    /// Default `false` keeps the legacy cold-start path so existing
    /// baselines stay bit-stable.
    pub warm_start: bool,
    /// **A1 — MPC-optimised footstep XY** (FullCentroidal MPC).
    ///
    /// When `true`, the FullCentroidal controller fills the per-leg
    /// touchdown XY target on the MPC's contact schedule (from the
    /// existing Raibert + cap-pt planner) and flips the MPC config's
    /// `q_foot_xy_world` to [`Self::q_foot_xy_world`]. The MPC adds
    /// a quadratic cost on the residual between its predicted foot
    /// landing and the planner target, letting it deviate the swing-
    /// leg joint trajectory to actively choose the footstep. This is
    /// the structural fix the P2 / `use_mpc_predicted_footstep`
    /// negative result identified as missing.
    ///
    /// Holding (cmd == 0) skips this block — there's no swing then.
    /// Default `false` keeps the legacy open-loop footstep regime.
    pub mpc_optimized_footstep: bool,
    /// Weight on the foot-XY tracking cost when
    /// [`Self::mpc_optimized_footstep`] is on. See
    /// [`crate::full_centroidal_mpc::FullCentroidalMpcConfig::q_foot_xy_world`].
    /// Default `500.0` — strong enough that a 1 cm landing error
    /// costs as much as a 1 N²·s² GRF deviation (`r_diag[GRF] = 1e-3`,
    /// so 1 N² ≡ 0.001; 0.01² · 500 = 0.05 ≫ 0.001) without
    /// drowning out the body-tracking terms.
    pub q_foot_xy_world: f64,
    /// **LinearCrawl-only**: fraction of each per-leg sub-cycle
    /// (`T/4`) held in 4-support before that leg lifts. Range `(0, 1)`;
    /// `0.5` is the default. Only [`crate::linear_crawl::LinearCrawlController`]
    /// reads this field — the body-velocity-tracking controllers
    /// (CHAMP / MPC / Centroidal / FullCentroidal) derive their support
    /// schedule from `gait_type.phase_offsets()` + `duty_factor` and
    /// ignore this knob, so changing it has no effect outside
    /// [`crate::GaitMode::LinearCrawl`].
    #[cfg_attr(feature = "serde", serde(default = "default_four_support_fraction"))]
    pub four_support_fraction: f64,
}

#[cfg(feature = "serde")]
fn default_four_support_fraction() -> f64 {
    0.5
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
            transition_fraction: 0.0,
            transition_enforce_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            mpc_optimized_footstep: false,
            q_foot_xy_world: 500.0,
            four_support_fraction: 0.5,
        }
    }

    /// 4-beat Walk preset. Same diagonal-sequence offsets as [`Self::crawl`]
    /// but a shorter cycle and a slightly lower duty (`0.75`) so the
    /// support polygon flips through three-leg / four-leg phases instead
    /// of always sitting on three. Reasonable speed/stability trade-off
    /// for a small quadruped.
    pub fn walk() -> Self {
        Self {
            gait_type: GaitType::Walk,
            cycle_period_s: 0.6,
            duty_factor: 0.75,
            swing_height_m: 0.035,
            max_step_length_m: 0.08,
            transition_fraction: 0.0,
            transition_enforce_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            mpc_optimized_footstep: false,
            q_foot_xy_world: 500.0,
            four_support_fraction: 0.5,
        }
    }

    /// Pace preset. Lateral pairs in phase (`FL+RL ↔ FR+RR`) — camel-
    /// style. Same timing/sizing as [`Self::trot`]; only the leg phasing
    /// differs. Tends to roll the trunk more than trot since both
    /// supporting legs are on the same side at a time.
    pub fn pace() -> Self {
        Self {
            gait_type: GaitType::Pace,
            cycle_period_s: 0.4,
            duty_factor: 0.5,
            swing_height_m: 0.04,
            max_step_length_m: 0.10,
            transition_fraction: 0.0,
            transition_enforce_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            mpc_optimized_footstep: false,
            q_foot_xy_world: 500.0,
            four_support_fraction: 0.5,
        }
    }

    /// Bound preset. Front pair vs rear pair anti-phase ("bunny hop").
    /// Shorter cycle, taller swing, larger stride than trot — fastest
    /// of the family. Not statically stable; relies on body inertia.
    pub fn bound() -> Self {
        Self {
            gait_type: GaitType::Bound,
            cycle_period_s: 0.3,
            duty_factor: 0.5,
            swing_height_m: 0.05,
            max_step_length_m: 0.12,
            transition_fraction: 0.0,
            transition_enforce_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            mpc_optimized_footstep: false,
            q_foot_xy_world: 500.0,
            four_support_fraction: 0.5,
        }
    }

    /// Dispatch by gait family — returns the canonical preset for `ty`.
    /// Use in GUIs / scripting where the user picks a [`GaitType`] and
    /// wants all the family-typical timing/sizing defaults at once
    /// rather than just the phase offsets.
    pub fn for_type(ty: GaitType) -> Self {
        match ty {
            GaitType::Trot => Self::trot(),
            GaitType::Walk => Self::walk(),
            GaitType::Pace => Self::pace(),
            GaitType::Bound => Self::bound(),
            GaitType::Crawl => Self::crawl(),
        }
    }

    /// Static crawl preset. Diagonal-sequence 4-beat with `duty=0.85`
    /// so swings never overlap (3 feet always on the ground), a long
    /// cycle so the body translates slowly, and a low swing height so
    /// the trunk barely pitches/rolls as each foot lifts. Use when you
    /// want the robot to walk straight forward with minimal sway —
    /// classical "creeping" gait.
    pub fn crawl() -> Self {
        Self {
            gait_type: GaitType::Crawl,
            // LinearCrawl-tuned defaults (validated on Go2 with vx=0.05
            // m/s, see `examples/go2_linear_crawl_sweep.rs`):
            //   T = 1.667 s, α = 0.85
            //     ⇒ t3 = 0.0625 s (3-leg support / swing),
            //       t4 = 0.3542 s (4-leg support)
            //   swing_h = 0.005 m ⇒ low trunk pitch/roll from leg lift
            // This combination gave the lowest |yaw|max and highest
            // forward grip in the sweep. Pair with actuator Kp ≈ 1000
            // for tightest body-velocity tracking.
            cycle_period_s: 1.667,
            duty_factor: 0.85,
            swing_height_m: 0.005,
            max_step_length_m: 0.06,
            transition_fraction: 0.0,
            transition_enforce_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            mpc_optimized_footstep: false,
            q_foot_xy_world: 500.0,
            four_support_fraction: 0.85,
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
    pub fn with_transition_fraction(mut self, tf: f64) -> Self {
        self.transition_fraction = tf.clamp(0.0, 0.5);
        self
    }
    pub fn with_transition_enforce_constraint(mut self, enable: bool) -> Self {
        self.transition_enforce_constraint = enable;
        self
    }
    /// LinearCrawl-only knob — see [`Self::four_support_fraction`].
    /// Clamped to `[0.05, 0.95]`.
    pub fn with_four_support_fraction(mut self, f: f64) -> Self {
        self.four_support_fraction = f.clamp(0.05, 0.95);
        self
    }
}

/// Compute the stance-leg GRF load weight at a given stance
/// `sub_fraction ∈ [0, 1]` (0 = just touched down, 1 = about to lift
/// off), given the `transition_fraction tw ∈ [0, 0.5]`.
///
/// Returns a weight in `[0, 1]`:
/// - Ramp up linearly from `0` → `1` over `[0, tw]`
/// - Hold at `1` over `[tw, 1 − tw]`
/// - Ramp down linearly from `1` → `0` over `[1 − tw, 1]`
///
/// `tw = 0` collapses to the legacy "always 1.0 in stance" behaviour.
/// `tw ≥ 0.5` clamps the hold region to a single point (peak at
/// `sub = 0.5`, ramping up over the first half and down over the
/// second half).
pub fn stance_weight_at(sub_fraction: f64, transition_fraction: f64) -> f64 {
    let s = sub_fraction.clamp(0.0, 1.0);
    let tw = transition_fraction.clamp(0.0, 0.5);
    if tw <= 0.0 {
        return 1.0;
    }
    let up = (s / tw).min(1.0);
    let down = ((1.0 - s) / tw).min(1.0);
    up.min(down).max(0.0)
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

/// Symmetric knee-bend pattern for the four legs, encoded as a two-character
/// shorthand. The first character is the front pair's bend direction, the
/// second is the rear pair's; left and right of each pair always match.
///
/// - `<` = knee bends backward (calf swings back from the knee)
/// - `>` = knee bends forward (calf swings forward from the knee)
///
/// So:
/// - `<<` (`BothBack`)        — every knee bends backward
/// - `<>` (`MammalianForward`) — front knees back, rear knees forward
///   (typical dog / horse layout viewed in profile: \\_/)
/// - `><` (`MammalianReverse`)— front forward, rear backward
///   (less common; some climbing robots)
/// - `>>` (`BothForward`)      — every knee forward
///
/// The pattern maps directly to the underlying `[bool; 4]` array indexed
/// `[FL, FR, RL, RR]` consumed by [`crate::ik::solve_leg_ik`]. Patterns
/// that aren't symmetric across the body's centerline aren't representable
/// — drop down to the per-leg `set_knee_forward` API for those.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum KneePattern {
    /// `<<` — every knee bends backward.
    BothBack,
    /// `<>` — front knees backward, rear knees forward (canonical mammal).
    MammalianForward,
    /// `><` — front knees forward, rear knees backward.
    MammalianReverse,
    /// `>>` — every knee bends forward.
    BothForward,
}

impl KneePattern {
    pub const ALL: [KneePattern; 4] = [
        KneePattern::BothBack,
        KneePattern::MammalianForward,
        KneePattern::MammalianReverse,
        KneePattern::BothForward,
    ];

    /// The shorthand string (`"<<"`, `"<>"`, `"><"`, `">>"`).
    pub fn label(self) -> &'static str {
        match self {
            KneePattern::BothBack => "<<",
            KneePattern::MammalianForward => "<>",
            KneePattern::MammalianReverse => "><",
            KneePattern::BothForward => ">>",
        }
    }

    /// Parse one of the four shorthand strings. Returns `None` for any
    /// other input so callers can detect typos.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "<<" => Some(KneePattern::BothBack),
            "<>" => Some(KneePattern::MammalianForward),
            "><" => Some(KneePattern::MammalianReverse),
            ">>" => Some(KneePattern::BothForward),
            _ => None,
        }
    }

    /// Convert to the per-leg knee-forward array indexed
    /// `[FL, FR, RL, RR]`. Both legs in a front/rear pair share the same
    /// boolean (no left/right asymmetry).
    pub fn to_knee_forward(self) -> [bool; 4] {
        match self {
            KneePattern::BothBack => [false, false, false, false],
            KneePattern::MammalianForward => [false, false, true, true],
            KneePattern::MammalianReverse => [true, true, false, false],
            KneePattern::BothForward => [true, true, true, true],
        }
    }

    /// Best-effort inverse of [`Self::to_knee_forward`]: compress an
    /// arbitrary `[FL, FR, RL, RR]` array into a symmetric pattern by
    /// looking only at the front/rear majorities. Asymmetric arrays
    /// (e.g. `[true, false, true, false]`) return whichever pattern's
    /// front/rear flags match the *first* member of each pair, so the
    /// round-trip via `to_knee_forward` may differ.
    pub fn from_knee_forward(arr: [bool; 4]) -> Self {
        match (arr[0], arr[2]) {
            (false, false) => KneePattern::BothBack,
            (false, true) => KneePattern::MammalianForward,
            (true, false) => KneePattern::MammalianReverse,
            (true, true) => KneePattern::BothForward,
        }
    }
}

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
    fn knee_pattern_label_round_trip() {
        for p in KneePattern::ALL {
            assert_eq!(KneePattern::from_label(p.label()), Some(p));
        }
        assert!(KneePattern::from_label("<<<").is_none());
        assert!(KneePattern::from_label("ab").is_none());
    }

    #[test]
    fn knee_pattern_to_array_symmetric() {
        // Each pattern maps to a left/right-symmetric [FL, FR, RL, RR].
        for p in KneePattern::ALL {
            let arr = p.to_knee_forward();
            assert_eq!(arr[0], arr[1], "pattern {:?} broke L/R front symmetry", p);
            assert_eq!(arr[2], arr[3], "pattern {:?} broke L/R rear symmetry", p);
        }
    }

    #[test]
    fn knee_pattern_specific_mappings() {
        // Pin down the specific bool layouts so a future refactor can't
        // silently flip the FL/FR/RL/RR slot order.
        assert_eq!(KneePattern::BothBack.to_knee_forward(), [false; 4]);
        assert_eq!(KneePattern::BothForward.to_knee_forward(), [true; 4]);
        assert_eq!(
            KneePattern::MammalianForward.to_knee_forward(),
            [false, false, true, true],
        );
        assert_eq!(
            KneePattern::MammalianReverse.to_knee_forward(),
            [true, true, false, false],
        );
    }

    #[test]
    fn knee_pattern_round_trips_via_array() {
        for p in KneePattern::ALL {
            let arr = p.to_knee_forward();
            assert_eq!(KneePattern::from_knee_forward(arr), p);
        }
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

    /// `transition_fraction = 0` collapses the weight to a constant
    /// `1.0` over the whole stance phase — backward-compat default.
    #[test]
    fn stance_weight_at_zero_transition_is_constant_one() {
        for k in 0..=10 {
            let s = k as f64 / 10.0;
            assert!((stance_weight_at(s, 0.0) - 1.0).abs() < 1e-12);
        }
    }

    /// `transition_fraction = 0.1`: weight ramps 0→1 over `[0, 0.1]`,
    /// is exactly 1 over `[0.1, 0.9]`, and ramps 1→0 over `[0.9, 1]`.
    /// The endpoints are exactly zero so the touchdown / lift-off
    /// step gets a zero GRF reference.
    #[test]
    fn stance_weight_at_ramps_at_boundaries() {
        // Touchdown edge.
        assert!((stance_weight_at(0.0, 0.1)).abs() < 1e-12);
        assert!((stance_weight_at(0.05, 0.1) - 0.5).abs() < 1e-12);
        assert!((stance_weight_at(0.10, 0.1) - 1.0).abs() < 1e-12);
        // Mid stance (well inside the hold band).
        assert!((stance_weight_at(0.5, 0.1) - 1.0).abs() < 1e-12);
        // Lift-off edge.
        assert!((stance_weight_at(0.90, 0.1) - 1.0).abs() < 1e-12);
        assert!((stance_weight_at(0.95, 0.1) - 0.5).abs() < 1e-12);
        assert!((stance_weight_at(1.00, 0.1)).abs() < 1e-12);
    }

    /// `transition_fraction = 0.5` is the degenerate maximum: the
    /// ramps meet at `sub = 0.5` and the hold region shrinks to a
    /// single point. Peak weight is 1.0 at `sub = 0.5`.
    #[test]
    fn stance_weight_at_clamps_at_half() {
        // tw = 0.6 is clamped down to 0.5 internally.
        assert!((stance_weight_at(0.5, 0.6) - 1.0).abs() < 1e-12);
        // Mirror symmetry across sub = 0.5.
        let a = stance_weight_at(0.25, 0.6);
        let b = stance_weight_at(0.75, 0.6);
        assert!((a - b).abs() < 1e-12);
        assert!((a - 0.5).abs() < 1e-12);
    }

    /// `sub_fraction` is clamped to `[0, 1]` — passing `1.5` or `-0.5`
    /// returns the same weight as `1.0` / `0.0`. Guards against the
    /// caller accidentally over-shooting the stance fraction.
    #[test]
    fn stance_weight_at_clamps_sub_fraction() {
        assert!((stance_weight_at(-0.5, 0.1) - stance_weight_at(0.0, 0.1)).abs() < 1e-12);
        assert!((stance_weight_at(1.5, 0.1) - stance_weight_at(1.0, 0.1)).abs() < 1e-12);
    }

    /// Crawl shares Walk's diagonal-sequence offsets but with a higher
    /// default duty factor — pin the contract so a future refactor
    /// can't silently change the lift order or the stability margin.
    #[test]
    fn crawl_phase_offsets_match_walk() {
        let crawl = GaitType::Crawl.phase_offsets();
        let walk = GaitType::Walk.phase_offsets();
        let mut c = std::collections::HashMap::new();
        let mut w = std::collections::HashMap::new();
        for (leg, off) in crawl {
            c.insert(leg, off);
        }
        for (leg, off) in walk {
            w.insert(leg, off);
        }
        for leg in [LegId::FL, LegId::FR, LegId::RL, LegId::RR] {
            assert_eq!(c[&leg], w[&leg], "crawl phase for {:?} != walk", leg);
        }
    }

    #[test]
    fn crawl_default_duty_is_static_stable() {
        // duty ≥ 0.75 + (small margin) ⇒ at most one foot in swing at a
        // time (three legs always supporting the body).
        let duty = GaitType::Crawl.default_duty_factor();
        assert!(duty > 0.75, "crawl duty {duty} would allow overlapping swings");
        assert!(duty < 1.0, "duty must leave some swing window");
    }

    /// `GaitConfig::crawl()` packages the preset the way the host app
    /// expects: smooth/slow body motion with low swing height.
    #[test]
    fn crawl_preset_is_slow_and_low() {
        let c = GaitConfig::crawl();
        assert_eq!(c.gait_type, GaitType::Crawl);
        assert!(c.cycle_period_s > GaitConfig::trot().cycle_period_s);
        assert!(c.duty_factor > 0.75);
        assert!(c.swing_height_m < GaitConfig::trot().swing_height_m);
        assert!(c.max_step_length_m < GaitConfig::trot().max_step_length_m);
    }

    /// `GaitConfig::for_type(t)` matches the direct preset constructor —
    /// pin this contract so consumers can route a `GaitType` selection
    /// through the dispatcher without subtle drift from the per-family
    /// builder.
    #[test]
    fn for_type_dispatches_to_each_preset() {
        for ty in GaitType::ALL {
            let cfg = GaitConfig::for_type(ty);
            assert_eq!(cfg.gait_type, ty);
            assert_eq!(cfg.duty_factor, ty.default_duty_factor());
        }
    }

    /// Each preset's `gait_type` matches the constructor name — guards
    /// against accidental copy/paste between `walk()` / `pace()` /
    /// `bound()` / `crawl()` (they share a lot of boilerplate).
    #[test]
    fn each_preset_carries_its_own_type() {
        assert_eq!(GaitConfig::trot().gait_type, GaitType::Trot);
        assert_eq!(GaitConfig::walk().gait_type, GaitType::Walk);
        assert_eq!(GaitConfig::pace().gait_type, GaitType::Pace);
        assert_eq!(GaitConfig::bound().gait_type, GaitType::Bound);
        assert_eq!(GaitConfig::crawl().gait_type, GaitType::Crawl);
    }

    /// Sanity bands so a future tweak doesn't silently swap Bound's
    /// "fast and bouncy" character for Walk's "slow and stable" one.
    #[test]
    fn preset_family_characters() {
        let crawl = GaitConfig::crawl();
        let walk = GaitConfig::walk();
        let trot = GaitConfig::trot();
        let bound = GaitConfig::bound();

        // Cycle ordering: bound < trot ≤ walk < crawl.
        assert!(bound.cycle_period_s < trot.cycle_period_s);
        assert!(trot.cycle_period_s <= walk.cycle_period_s);
        assert!(walk.cycle_period_s < crawl.cycle_period_s);

        // Duty ordering: bound = trot < walk < crawl.
        assert!(bound.duty_factor < walk.duty_factor);
        assert!(walk.duty_factor < crawl.duty_factor);

        // Bound is the highest-swing & longest-stride preset.
        assert!(bound.swing_height_m > trot.swing_height_m);
        assert!(bound.max_step_length_m > trot.max_step_length_m);

        // Crawl is the lowest-swing & shortest-stride preset.
        assert!(crawl.swing_height_m < trot.swing_height_m);
        assert!(crawl.max_step_length_m < trot.max_step_length_m);
    }

    /// Walk-and-Crawl share offsets but every leg has a *distinct* phase
    /// — that's what makes them 4-beat. (Trot/Pace/Bound are 2-beat.)
    #[test]
    fn crawl_is_four_beat() {
        let offsets = GaitType::Crawl.phase_offsets();
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (_leg, off) in offsets {
            seen.insert((off * 1000.0) as i64);
        }
        assert_eq!(seen.len(), 4, "crawl must use four distinct phase offsets");
    }
}
