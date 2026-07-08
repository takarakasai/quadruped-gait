//! Metadata-driven surface for the experimental research knobs.
//!
//! Every research toggle (the C1/A1/A3/B3/P2… bench experiments) is
//! described by an [`ExpKey`] — name, widget kind / range, and the
//! bench summary as help text — and read / written through
//! [`AnyGaitController::get_experimental`] /
//! [`AnyGaitController::set_experimental`](crate::AnyGaitController::set_experimental).
//! Hosts render their "Experimental flags" UI *from this table*, so
//! adding a new experiment is a change to this crate only: define the
//! knob here, wire it in `get/set_experimental`, and every host picks
//! it up without a code change.
//!
//! Scope: only research knobs live here. The stable driving API
//! (mode, velocity command, configs, observed state) stays typed.

use crate::generator::AnyGaitController;
use crate::GaitGenerator as _;

/// Value of an experimental knob.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExpValue {
    Bool(bool),
    F64(f64),
}

impl ExpValue {
    pub fn as_bool(self) -> Option<bool> {
        match self {
            ExpValue::Bool(b) => Some(b),
            ExpValue::F64(_) => None,
        }
    }
    pub fn as_f64(self) -> Option<f64> {
        match self {
            ExpValue::F64(v) => Some(v),
            ExpValue::Bool(_) => None,
        }
    }
}

/// Widget kind (and range) for an experimental knob.
#[derive(Clone, Copy, Debug)]
pub enum ExpKind {
    /// Render as a checkbox.
    Bool,
    /// Render as a slider over `[min, max]`.
    F64 {
        min: f64,
        max: f64,
        /// Logarithmic slider scale (for penalty / weight knobs).
        logarithmic: bool,
        /// Fixed decimals to display.
        decimals: u8,
    },
}

/// One experimental knob: identity, widget metadata, and the bench
/// summary a host shows as hover help.
#[derive(Clone, Copy, Debug)]
pub struct ExpKey {
    /// Stable identifier, used with `get/set_experimental`.
    pub key: &'static str,
    /// Short UI label.
    pub label: &'static str,
    pub kind: ExpKind,
    /// What the knob does and what the bench showed. This is the
    /// single home for the experiment's summary — keep it current.
    pub help: &'static str,
}

/// Error from [`AnyGaitController::set_experimental`].
#[derive(Clone, Debug, PartialEq)]
pub enum ExpError {
    /// The key is not defined for the active controller mode.
    UnknownKey(String),
    /// The key exists but the value variant doesn't match its kind.
    WrongKind { key: &'static str },
}

impl std::fmt::Display for ExpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpError::UnknownKey(k) => write!(f, "unknown experimental key '{k}'"),
            ExpError::WrongKind { key } => {
                write!(f, "wrong value kind for experimental key '{key}'")
            }
        }
    }
}

impl std::error::Error for ExpError {}

/// Experimental knobs of the FullCentroidal MPC controller, in the
/// order hosts should render them.
pub const FULL_CENTROIDAL_EXP_KEYS: &[ExpKey] = &[
    ExpKey {
        key: "legged_control_parity",
        label: "legged_control parity",
        kind: ExpKind::Bool,
        help: "Per-step phase projection + NormalVelocityConstraintCppAd \
               analogue (swing leg vertical foot velocity equality). \
               Bench note: didn't fix lateral 4N+ fall on its own \
               (cap-pt 0.05 did). Kept for A/B and as the prerequisite \
               for transition_fraction below.",
    },
    ExpKey {
        key: "parity_use_nominal_q_ref",
        label: "parity: nominal q_ref",
        kind: ExpKind::Bool,
        help: "Parity sub-flag: use the URDF nominal pose as the joint_q \
               tracking reference (legged_control DEFAULT_JOINT_STATE \
               analogue) instead of the IK-projected reference. Only \
               effective while parity is on.",
    },
    ExpKey {
        key: "transition_fraction",
        label: "transition_fraction",
        kind: ExpKind::F64 { min: 0.0, max: 0.30, logarithmic: false, decimals: 2 },
        help: "C1 experiment: ramps the per-leg GRF reference at touchdown / \
               lift-off. By itself (cost-side) bench was bit-exact identical \
               to off — `r_diag[GRF]` is too small to make the MPC track the \
               ramp. Pair with the constraint-side toggle for the real effect.",
    },
    ExpKey {
        key: "transition_enforce_constraint",
        label: "transition: enforce as hard constraint (C1-2)",
        kind: ExpKind::Bool,
        help: "C1-2: ramps the per-leg `max_normal_force` upper bound at \
               touchdown / lift-off as a HARD QP inequality. Bench: \
               lateral 6N peak roll −30 %, forward 6N peak |dy| −42 % \
               at trans_fraction = 0.05. Off when transition_fraction = 0.",
    },
    ExpKey {
        key: "friction_cone_soft",
        label: "friction cone soft + slack (A3)",
        kind: ExpKind::Bool,
        help: "A3: relaxes the friction pyramid via per-(leg, step) slack \
               variables `s_x, s_y ≥ 0` with quadratic penalty `λ · s²`. \
               Useful at the pyramid corner (μ=0.5 lateral 4-6N regime) \
               where the hard form returns AlmostSolved or falls back to \
               the reference. f_z bounds stay hard. legged_control \
               analogue: FrictionConeConstraint + RelaxedBarrierPenalty.",
    },
    ExpKey {
        key: "friction_cone_slack_penalty",
        label: "slack penalty",
        kind: ExpKind::F64 { min: 10.0, max: 10_000.0, logarithmic: true, decimals: 0 },
        help: "Quadratic cost on each `s_i`. Larger → cone stays closer to \
               hard. Smaller → more slack budget under disturbance. Only \
               effective when A3 is on.",
    },
    ExpKey {
        key: "warm_start",
        label: "MPC warm-start (B3)",
        kind: ExpKind::Bool,
        help: "B3: seed each MPC tick's SQP iter 0 from the previous tick's \
               solved trajectory (shifted by one step) instead of the \
               gravity-balanced cmd reference. Same convergence point at \
               steady state, but fewer iterations to get there — typical \
               2× speed-up on cmd-held workloads. legged_control analogue: \
               OCS2's solverObservation warm-start.",
    },
    ExpKey {
        key: "mpc_optimized_footstep",
        label: "MPC-optimised footstep XY (A1)",
        kind: ExpKind::Bool,
        help: "A1: adds a soft cost penalising the predicted foot-XY vs the \
               planner-supplied touchdown target. The MPC deviates the \
               swing-leg joint trajectory to land at the target, \
               self-consistently with its predicted base motion. Closes \
               the loop that P2 couldn't.",
    },
    ExpKey {
        key: "q_foot_xy_world",
        label: "q_foot_xy_world",
        kind: ExpKind::F64 { min: 10.0, max: 5_000.0, logarithmic: true, decimals: 0 },
        help: "Weight on the foot-XY tracking residual. Only active when A1 \
               is on. Higher → more aggressive footstep tracking, may \
               overshoot on jumpy planner targets.",
    },
    ExpKey {
        key: "use_mpc_predicted_footstep",
        label: "MPC-predicted footstep (P2)",
        kind: ExpKind::Bool,
        help: "Replaces cap-pt feedback with a footstep correction derived \
               from the MPC's predicted base trajectory (legged_control \
               SwingTrajectoryPlanner analogue). Bench: **made lateral push \
               worse** because without A1 the MPC doesn't optimise foot \
               XY — the predicted base reflects sliding, not restoring. \
               Kept as a documented negative result.",
    },
];

impl AnyGaitController {
    /// The experimental knobs applicable to the active controller, in
    /// render order. Empty for modes without research knobs.
    pub fn experimental_keys(&self) -> &'static [ExpKey] {
        match self {
            AnyGaitController::FullCentroidal(_) => FULL_CENTROIDAL_EXP_KEYS,
            _ => &[],
        }
    }

    /// Read an experimental knob. `None` if the key doesn't apply to
    /// the active controller mode.
    pub fn get_experimental(&self, key: &str) -> Option<ExpValue> {
        let AnyGaitController::FullCentroidal(c) = self else {
            return None;
        };
        let cfg = self.config();
        Some(match key {
            "legged_control_parity" => ExpValue::Bool(c.legged_control_parity()),
            "parity_use_nominal_q_ref" => ExpValue::Bool(c.parity_use_nominal_q_ref()),
            "use_mpc_predicted_footstep" => ExpValue::Bool(c.use_mpc_predicted_footstep()),
            "transition_fraction" => ExpValue::F64(cfg.transition_fraction),
            "transition_enforce_constraint" => {
                ExpValue::Bool(cfg.transition_enforce_constraint)
            }
            "friction_cone_soft" => ExpValue::Bool(cfg.friction_cone_soft),
            "friction_cone_slack_penalty" => ExpValue::F64(cfg.friction_cone_slack_penalty),
            "warm_start" => ExpValue::Bool(cfg.warm_start),
            "mpc_optimized_footstep" => ExpValue::Bool(cfg.mpc_optimized_footstep),
            "q_foot_xy_world" => ExpValue::F64(cfg.q_foot_xy_world),
            _ => return None,
        })
    }

    /// Write an experimental knob on the active controller.
    ///
    /// Config-backed knobs go through the same clone-config →
    /// `set_config` path the hosts used to hand-write, so the
    /// controller reacts exactly as before.
    pub fn set_experimental(&mut self, key: &str, value: ExpValue) -> Result<(), ExpError> {
        if !self.experimental_keys().iter().any(|k| k.key == key) {
            return Err(ExpError::UnknownKey(key.to_string()));
        }
        let wrong = |key: &'static str| ExpError::WrongKind { key };
        match key {
            "legged_control_parity" => {
                self.set_legged_control_parity(
                    value.as_bool().ok_or(wrong("legged_control_parity"))?,
                );
            }
            "parity_use_nominal_q_ref" => {
                self.set_parity_use_nominal_q_ref(
                    value.as_bool().ok_or(wrong("parity_use_nominal_q_ref"))?,
                );
            }
            "use_mpc_predicted_footstep" => {
                self.set_use_mpc_predicted_footstep(
                    value.as_bool().ok_or(wrong("use_mpc_predicted_footstep"))?,
                );
            }
            _ => {
                let mut cfg = self.config().clone();
                match key {
                    "transition_fraction" => {
                        cfg.transition_fraction =
                            value.as_f64().ok_or(wrong("transition_fraction"))?;
                    }
                    "transition_enforce_constraint" => {
                        cfg.transition_enforce_constraint = value
                            .as_bool()
                            .ok_or(wrong("transition_enforce_constraint"))?;
                    }
                    "friction_cone_soft" => {
                        cfg.friction_cone_soft =
                            value.as_bool().ok_or(wrong("friction_cone_soft"))?;
                    }
                    "friction_cone_slack_penalty" => {
                        cfg.friction_cone_slack_penalty =
                            value.as_f64().ok_or(wrong("friction_cone_slack_penalty"))?;
                    }
                    "warm_start" => {
                        cfg.warm_start = value.as_bool().ok_or(wrong("warm_start"))?;
                    }
                    "mpc_optimized_footstep" => {
                        cfg.mpc_optimized_footstep =
                            value.as_bool().ok_or(wrong("mpc_optimized_footstep"))?;
                    }
                    "q_foot_xy_world" => {
                        cfg.q_foot_xy_world =
                            value.as_f64().ok_or(wrong("q_foot_xy_world"))?;
                    }
                    _ => unreachable!("key checked against experimental_keys above"),
                }
                self.set_config(cfg);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{KinematicsConfig, LegKinematics};
    use crate::{GaitConfig, GaitMode, LegId};
    use nalgebra::Vector3;

    fn build_kin() -> KinematicsConfig {
        let mk = |leg: LegId, sx: f64, sy: f64, prefix: &str| {
            let mut k = LegKinematics::new(
                leg,
                format!("{prefix}_hip"),
                format!("{prefix}_thigh"),
                format!("{prefix}_calf"),
                format!("{prefix}_foot"),
                Vector3::new(sx * 0.18, sy * 0.05, 0.0),
                0.04,
                0.18,
                0.18,
            );
            k.nominal_foot_body.z = k.hip_offset.z - 0.36 * 0.92;
            k
        };
        KinematicsConfig {
            fl: mk(LegId::FL, 1.0, 1.0, "FL"),
            fr: mk(LegId::FR, 1.0, -1.0, "FR"),
            rl: mk(LegId::RL, -1.0, 1.0, "RL"),
            rr: mk(LegId::RR, -1.0, -1.0, "RR"),
        }
    }

    pub(super) fn build_fullc() -> AnyGaitController {
        AnyGaitController::new(GaitMode::FullCentroidal, GaitConfig::trot(), build_kin())
    }

    #[test]
    fn keys_roundtrip_get_set() {
        let mut c = build_fullc();
        for k in c.experimental_keys() {
            let cur = c.get_experimental(k.key).expect("declared key must read");
            // Write a different value, read it back.
            let new = match cur {
                ExpValue::Bool(b) => ExpValue::Bool(!b),
                ExpValue::F64(v) => ExpValue::F64(v + 1.0),
            };
            c.set_experimental(k.key, new).expect("declared key must write");
            assert_eq!(c.get_experimental(k.key), Some(new), "key {}", k.key);
        }
    }

    #[test]
    fn unknown_key_and_wrong_kind_error() {
        let mut c = build_fullc();
        assert!(matches!(
            c.set_experimental("no_such_knob", ExpValue::Bool(true)),
            Err(ExpError::UnknownKey(_))
        ));
        assert!(matches!(
            c.set_experimental("warm_start", ExpValue::F64(1.0)),
            Err(ExpError::WrongKind { key: "warm_start" })
        ));
    }

    #[test]
    fn non_fullc_modes_expose_no_keys() {
        let c = AnyGaitController::new(GaitMode::Champ, GaitConfig::trot(), build_kin());
        assert!(c.experimental_keys().is_empty());
        assert_eq!(c.get_experimental("warm_start"), None);
    }
}

// ═════════════════════════ Presets ══════════════════════════════════════

/// A named snapshot of experimental-knob values.
///
/// Presets make a tuned knob combination reproducible across sessions
/// and hosts: the GUI saves the current Experimental section under a
/// name, and the headless sweeper / runner recalls it by name. Values
/// are stored per key so a preset written against one controller
/// version applies cleanly to a newer one (unknown keys are reported,
/// not fatal).
#[derive(Clone, Debug, PartialEq)]
pub struct ExpPreset {
    pub name: String,
    /// `(key, value)` pairs in file order.
    pub values: Vec<(String, ExpValue)>,
}

impl AnyGaitController {
    /// Snapshot the current experimental-knob values as a preset.
    pub fn snapshot_experimental(&self, name: impl Into<String>) -> ExpPreset {
        let values = self
            .experimental_keys()
            .iter()
            .filter_map(|k| self.get_experimental(k.key).map(|v| (k.key.to_string(), v)))
            .collect();
        ExpPreset {
            name: name.into(),
            values,
        }
    }

    /// Apply a preset to the active controller. Returns the number of
    /// knobs applied plus the keys that were skipped (unknown to the
    /// current mode / controller version, or wrong value kind) — the
    /// caller decides whether skips are worth surfacing.
    pub fn apply_experimental(&mut self, preset: &ExpPreset) -> (usize, Vec<String>) {
        let mut applied = 0;
        let mut skipped = Vec::new();
        for (key, value) in &preset.values {
            match self.set_experimental(key, *value) {
                Ok(()) => applied += 1,
                Err(_) => skipped.push(key.clone()),
            }
        }
        (applied, skipped)
    }
}

/// Replace the preset with the same name, or append. Keeps file order
/// stable for the existing entries.
pub fn upsert_preset(presets: &mut Vec<ExpPreset>, preset: ExpPreset) {
    match presets.iter_mut().find(|p| p.name == preset.name) {
        Some(slot) => *slot = preset,
        None => presets.push(preset),
    }
}

/// Serialise presets to the on-disk text form.
///
/// The format is a TOML subset kept hand-parsable so this crate stays
/// dependency-free:
///
/// ```text
/// # quadruped-gait experimental presets
/// [lateral-robust]
/// transition_fraction = 0.05
/// transition_enforce_constraint = true
/// ```
///
/// One `[section]` per preset; each line is `key = true|false|<f64>`.
pub fn format_presets(presets: &[ExpPreset]) -> String {
    let mut s = String::from("# quadruped-gait experimental presets\n");
    for p in presets {
        s.push_str(&format!("\n[{}]\n", p.name));
        for (key, value) in &p.values {
            match value {
                ExpValue::Bool(b) => s.push_str(&format!("{key} = {b}\n")),
                ExpValue::F64(v) => {
                    // Keep integral values readable but unambiguously
                    // float-typed on re-parse (`1.0`, not `1`).
                    if v.fract() == 0.0 && v.is_finite() {
                        s.push_str(&format!("{key} = {v:.1}\n"));
                    } else {
                        s.push_str(&format!("{key} = {v}\n"));
                    }
                }
            }
        }
    }
    s
}

/// Parse the text form produced by [`format_presets`] (see there for
/// the grammar). Unknown *keys* are kept — validity against a concrete
/// controller is [`AnyGaitController::apply_experimental`]'s job — but
/// malformed *syntax* is an error with a line number.
pub fn parse_presets(text: &str) -> Result<Vec<ExpPreset>, String> {
    let mut presets: Vec<ExpPreset> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            let name = name.trim();
            if name.is_empty() {
                return Err(format!("line {}: empty preset name", lineno + 1));
            }
            presets.push(ExpPreset {
                name: name.to_string(),
                values: Vec::new(),
            });
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {}: expected 'key = value': {raw}", lineno + 1));
        };
        let Some(preset) = presets.last_mut() else {
            return Err(format!(
                "line {}: key/value before any [preset] section",
                lineno + 1
            ));
        };
        let key = key.trim().to_string();
        let value = value.trim();
        let value = match value {
            "true" => ExpValue::Bool(true),
            "false" => ExpValue::Bool(false),
            v => ExpValue::F64(v.parse::<f64>().map_err(|e| {
                format!("line {}: bad value '{v}' for '{key}': {e}", lineno + 1)
            })?),
        };
        preset.values.push((key, value));
    }
    Ok(presets)
}

/// Load presets from a file. A missing file is an empty preset list
/// (the natural first-run state), not an error.
pub fn load_presets(path: &std::path::Path) -> Result<Vec<ExpPreset>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_presets(&text)
}

/// Save presets to a file (overwrites).
pub fn save_presets(path: &std::path::Path, presets: &[ExpPreset]) -> Result<(), String> {
    std::fs::write(path, format_presets(presets))
        .map_err(|e| format!("write {}: {e}", path.display()))
}

#[cfg(test)]
mod preset_tests {
    use super::tests::build_fullc;
    use super::*;

    #[test]
    fn presets_round_trip_through_text() {
        let presets = vec![
            ExpPreset {
                name: "lateral-robust".into(),
                values: vec![
                    ("transition_fraction".into(), ExpValue::F64(0.05)),
                    ("transition_enforce_constraint".into(), ExpValue::Bool(true)),
                    ("friction_cone_slack_penalty".into(), ExpValue::F64(1000.0)),
                ],
            },
            ExpPreset {
                name: "baseline".into(),
                values: vec![("warm_start".into(), ExpValue::Bool(false))],
            },
        ];
        let text = format_presets(&presets);
        let back = parse_presets(&text).unwrap();
        assert_eq!(back, presets);
    }

    #[test]
    fn parse_rejects_malformed_lines() {
        assert!(parse_presets("[ok]\nnot a kv line").is_err());
        assert!(parse_presets("orphan = 1.0").is_err());
        assert!(parse_presets("[]\n").is_err());
        assert!(parse_presets("[ok]\nkey = maybe").is_err());
    }

    #[test]
    fn snapshot_apply_round_trip_on_controller() {
        let mut c = build_fullc();
        c.set_experimental("transition_fraction", ExpValue::F64(0.05))
            .unwrap();
        c.set_experimental("transition_enforce_constraint", ExpValue::Bool(true))
            .unwrap();
        let preset = c.snapshot_experimental("tuned");
        assert!(!preset.values.is_empty());

        // Fresh controller: apply and compare every knob.
        let mut fresh = build_fullc();
        let (applied, skipped) = fresh.apply_experimental(&preset);
        assert_eq!(applied, preset.values.len());
        assert_eq!(skipped, Vec::<String>::new());
        for (key, value) in &preset.values {
            assert_eq!(fresh.get_experimental(key), Some(*value), "key {key}");
        }
    }

    #[test]
    fn apply_reports_unknown_keys_without_failing() {
        let mut c = build_fullc();
        let preset = ExpPreset {
            name: "future".into(),
            values: vec![
                ("warm_start".into(), ExpValue::Bool(true)),
                ("knob_from_the_future".into(), ExpValue::F64(1.0)),
            ],
        };
        let (applied, skipped) = c.apply_experimental(&preset);
        assert_eq!(applied, 1);
        assert_eq!(skipped, vec!["knob_from_the_future".to_string()]);
    }

    #[test]
    fn upsert_replaces_by_name() {
        let mut v = vec![ExpPreset { name: "a".into(), values: vec![] }];
        upsert_preset(&mut v, ExpPreset {
            name: "a".into(),
            values: vec![("warm_start".into(), ExpValue::Bool(true))],
        });
        upsert_preset(&mut v, ExpPreset { name: "b".into(), values: vec![] });
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].values.len(), 1);
    }
}
