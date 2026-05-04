//! Pluggable gait-generator interface so callers can switch between the
//! CHAMP-derived open-loop controller and the MPC-flavoured closed-loop
//! controller without rewriting their integration glue.
//!
//! Two implementations live in this crate:
//!
//! - [`crate::ChampGaitController`] — the classic Raibert-heuristic /
//!   Bezier-swing pipeline. Open-loop in body frame, hands joint angle
//!   targets to a position-controlled actuator stack. Same code path
//!   articara has shipped since v0.
//! - [`crate::MpcGaitController`] — adds the **capture-point feedback**
//!   term (`√(h/g) · (v_actual − v_cmd)`) to the Raibert footstep
//!   formula plus a multi-step LIP horizon for the next swing target,
//!   making the gait closed-loop on the host's reported body velocity.
//!
//! Both produce the same [`ControllerOutput`] shape so the consumer
//! (rendering, MuJoCo position-target dispatch, gait-panel UI) is
//! oblivious to which one is active. [`GaitGenerator`] is the trait,
//! [`AnyGaitController`] is an enum-dispatched wrapper articara uses to
//! avoid the `Box<dyn>` overhead in the sim hot path.

use nalgebra::Vector3;

use crate::config::{GaitConfig, KinematicsConfig, KneePattern, LegId, VelocityCmd};
use crate::controller::{ControllerOutput, GaitController as ChampGaitController};

/// Common interface every gait generator implements.
///
/// Methods in the **observation** group ([`Self::set_body_state_observed`])
/// are no-ops for open-loop controllers ([`ChampGaitController`]) and
/// drive the feedback term in closed-loop ones
/// ([`crate::MpcGaitController`]). Hosts can call them
/// unconditionally without checking the active mode.
pub trait GaitGenerator {
    /// Advance the gait by `dt` seconds; return the per-leg outputs
    /// the host should hand to the actuator stack.
    fn tick(&mut self, dt: f64) -> ControllerOutput;

    /// Set the desired body velocity (vx, vy, wz). Persists until the
    /// next call.
    fn set_velocity_cmd(&mut self, cmd: VelocityCmd);

    fn velocity_cmd(&self) -> VelocityCmd;

    /// Reset internal state (phase + integrated body pose) to the
    /// origin and zero the velocity command.
    fn reset(&mut self);

    fn config(&self) -> &GaitConfig;
    fn set_config(&mut self, cfg: GaitConfig);

    fn kinematics(&self) -> &KinematicsConfig;

    fn set_knee_forward(&mut self, leg: LegId, forward: bool);
    fn set_knee_pattern(&mut self, pattern: KneePattern);
    fn knee_pattern(&self) -> KneePattern;
    fn knee_forward(&self) -> [bool; 4];

    /// Feed the **observed** body state (linear velocity in world
    /// frame) so closed-loop generators can compute the capture-point
    /// feedback term. Default impl ignores it — open-loop controllers
    /// don't need it.
    fn set_body_state_observed(
        &mut self,
        _world_linear_velocity: Vector3<f64>,
    ) {
    }
}

// ─── Trait impl for the existing CHAMP controller ─────────────────────

impl GaitGenerator for ChampGaitController {
    fn tick(&mut self, dt: f64) -> ControllerOutput {
        ChampGaitController::tick(self, dt)
    }
    fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        ChampGaitController::set_velocity_cmd(self, cmd);
    }
    fn velocity_cmd(&self) -> VelocityCmd {
        ChampGaitController::velocity_cmd(self)
    }
    fn reset(&mut self) {
        ChampGaitController::reset(self);
    }
    fn config(&self) -> &GaitConfig {
        ChampGaitController::config(self)
    }
    fn set_config(&mut self, cfg: GaitConfig) {
        ChampGaitController::set_config(self, cfg);
    }
    fn kinematics(&self) -> &KinematicsConfig {
        ChampGaitController::kinematics(self)
    }
    fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        ChampGaitController::set_knee_forward(self, leg, forward);
    }
    fn set_knee_pattern(&mut self, pattern: KneePattern) {
        ChampGaitController::set_knee_pattern(self, pattern);
    }
    fn knee_pattern(&self) -> KneePattern {
        ChampGaitController::knee_pattern(self)
    }
    fn knee_forward(&self) -> [bool; 4] {
        ChampGaitController::knee_forward(self)
    }
    // CHAMP is open-loop → keep the default no-op impl.
}

// ─── Mode tag + enum dispatch wrapper ─────────────────────────────────

/// Which generator is active. Persisted by the host (`articara` stores
/// it in app state and in `.misa` `[[gait]]` entries via
/// [`crate::config::GaitConfig`]) so the choice survives restarts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GaitMode {
    /// CHAMP-derived open-loop controller (default).
    Champ,
    /// MPC-flavoured closed-loop controller (LIP capture-point
    /// feedback + multi-step horizon).
    Mpc,
}

impl Default for GaitMode {
    fn default() -> Self {
        Self::Champ
    }
}

impl GaitMode {
    /// Short label for UI dropdowns.
    pub fn label(self) -> &'static str {
        match self {
            GaitMode::Champ => "CHAMP (open-loop)",
            GaitMode::Mpc => "MPC (capture-point)",
        }
    }
    pub const ALL: [GaitMode; 2] = [GaitMode::Champ, GaitMode::Mpc];
}

/// Enum-dispatched wrapper used by `articara` to avoid a `Box<dyn
/// GaitGenerator>` allocation in the per-tick hot path. The variants
/// share the same construction inputs ([`GaitConfig`] +
/// [`KinematicsConfig`]) so switching modes mid-session preserves the
/// model.
pub enum AnyGaitController {
    Champ(ChampGaitController),
    Mpc(crate::MpcGaitController),
}

impl AnyGaitController {
    pub fn new(mode: GaitMode, cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        match mode {
            GaitMode::Champ => {
                AnyGaitController::Champ(ChampGaitController::new(cfg, kin))
            }
            GaitMode::Mpc => {
                AnyGaitController::Mpc(crate::MpcGaitController::new(cfg, kin))
            }
        }
    }

    pub fn mode(&self) -> GaitMode {
        match self {
            AnyGaitController::Champ(_) => GaitMode::Champ,
            AnyGaitController::Mpc(_) => GaitMode::Mpc,
        }
    }

    /// Switch the active controller while preserving `cmd`, `cfg`, and
    /// `knee_forward`. Phase + body-pose integrators are re-initialised
    /// because the two controllers don't share that state representation.
    pub fn set_mode(&mut self, mode: GaitMode) {
        if mode == self.mode() {
            return;
        }
        let cmd = self.velocity_cmd();
        let cfg = self.config().clone();
        let kin = self.kinematics().clone();
        let knee = self.knee_forward();
        let mut new = AnyGaitController::new(mode, cfg, kin);
        new.set_velocity_cmd(cmd);
        for (i, fwd) in knee.iter().enumerate() {
            new.set_knee_forward(LegId::ALL[i], *fwd);
        }
        *self = new;
    }
}

impl AnyGaitController {
    /// SRBD MPC predicted ground reaction forces from the last tick,
    /// when the active mode is [`GaitMode::Mpc`]. CHAMP variant
    /// always returns `None` (it has no MPC state).
    pub fn predicted_grfs(&self) -> Option<&crate::srbd_mpc::MpcSolution> {
        match self {
            AnyGaitController::Champ(_) => None,
            AnyGaitController::Mpc(c) => c.predicted_grfs(),
        }
    }

    /// Per-leg stance-foot torque feedforward (`τ = -J^T·f_GRF`)
    /// computed from the last MPC solve. See
    /// [`crate::MpcGaitController::stance_grf_torques`] for the
    /// formulation. Returns `[None; 4]` when the active mode is CHAMP
    /// or no solution is available yet — the caller can treat this
    /// uniformly as "no torque feedforward".
    pub fn stance_grf_torques(
        &self,
        output: &ControllerOutput,
    ) -> [Option<[f64; 3]>; 4] {
        match self {
            AnyGaitController::Champ(_) => [None; 4],
            AnyGaitController::Mpc(c) => c.stance_grf_torques(output),
        }
    }
}

impl GaitGenerator for AnyGaitController {
    fn tick(&mut self, dt: f64) -> ControllerOutput {
        match self {
            AnyGaitController::Champ(c) => c.tick(dt),
            AnyGaitController::Mpc(c) => c.tick(dt),
        }
    }
    fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        match self {
            AnyGaitController::Champ(c) => c.set_velocity_cmd(cmd),
            AnyGaitController::Mpc(c) => c.set_velocity_cmd(cmd),
        }
    }
    fn velocity_cmd(&self) -> VelocityCmd {
        match self {
            AnyGaitController::Champ(c) => c.velocity_cmd(),
            AnyGaitController::Mpc(c) => c.velocity_cmd(),
        }
    }
    fn reset(&mut self) {
        match self {
            AnyGaitController::Champ(c) => c.reset(),
            AnyGaitController::Mpc(c) => c.reset(),
        }
    }
    fn config(&self) -> &GaitConfig {
        match self {
            AnyGaitController::Champ(c) => c.config(),
            AnyGaitController::Mpc(c) => c.config(),
        }
    }
    fn set_config(&mut self, cfg: GaitConfig) {
        match self {
            AnyGaitController::Champ(c) => c.set_config(cfg),
            AnyGaitController::Mpc(c) => c.set_config(cfg),
        }
    }
    fn kinematics(&self) -> &KinematicsConfig {
        match self {
            AnyGaitController::Champ(c) => c.kinematics(),
            AnyGaitController::Mpc(c) => c.kinematics(),
        }
    }
    fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        match self {
            AnyGaitController::Champ(c) => c.set_knee_forward(leg, forward),
            AnyGaitController::Mpc(c) => c.set_knee_forward(leg, forward),
        }
    }
    fn set_knee_pattern(&mut self, pattern: KneePattern) {
        match self {
            AnyGaitController::Champ(c) => c.set_knee_pattern(pattern),
            AnyGaitController::Mpc(c) => c.set_knee_pattern(pattern),
        }
    }
    fn knee_pattern(&self) -> KneePattern {
        match self {
            AnyGaitController::Champ(c) => c.knee_pattern(),
            AnyGaitController::Mpc(c) => c.knee_pattern(),
        }
    }
    fn knee_forward(&self) -> [bool; 4] {
        match self {
            AnyGaitController::Champ(c) => c.knee_forward(),
            AnyGaitController::Mpc(c) => c.knee_forward(),
        }
    }
    fn set_body_state_observed(&mut self, v_world: Vector3<f64>) {
        match self {
            AnyGaitController::Champ(c) => c.set_body_state_observed(v_world),
            AnyGaitController::Mpc(c) => c.set_body_state_observed(v_world),
        }
    }
}
