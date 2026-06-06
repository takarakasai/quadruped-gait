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
    fn set_kinematics(&mut self, kin: KinematicsConfig);

    fn set_knee_forward(&mut self, leg: LegId, forward: bool);
    fn set_knee_pattern(&mut self, pattern: KneePattern);
    fn knee_pattern(&self) -> KneePattern;
    fn knee_forward(&self) -> [bool; 4];

    /// Feed the **observed** body state (linear + angular velocity in
    /// world frame) so closed-loop generators can compute the capture-
    /// point feedback term and the SRBD MPC's angular tracking. Default
    /// impl ignores it — open-loop controllers don't need it.
    fn set_body_state_observed(
        &mut self,
        _world_linear_velocity: Vector3<f64>,
        _world_angular_velocity: Vector3<f64>,
    ) {
    }

    /// Feed the **observed** body pose (world-frame yaw + position) so
    /// closed-loop generators can replace their command-integrated
    /// `body_state` with the real one. Default impl ignores — open-loop
    /// controllers don't reference body_state externally.
    fn set_body_pose_observed(
        &mut self,
        _world_yaw: f64,
        _world_position: Vector3<f64>,
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
    fn set_kinematics(&mut self, kin: KinematicsConfig) {
        ChampGaitController::set_kinematics(self, kin);
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
    /// MPC-flavoured closed-loop controller using the body-root SRBD
    /// model (Di Carlo 2018, LIP capture-point + multi-step horizon).
    Mpc,
    /// Centroidal-SRBD MPC: same convex-QP family as `Mpc` but with
    /// the state expressed in centroidal momentum coordinates so the
    /// robot's CoM offset is modelled exactly. Equivalent to
    /// legged_control's `centroidalModelType = 1` (Single Rigid Body
    /// over centroidal coordinates). Available for direct comparison
    /// against `Mpc` — both are kept as baselines.
    CentroidalSrbd,
    /// Full-centroidal MPC (D3.3): 25-state SQP with joint angles in
    /// the state and joint velocities in the input, plus a stance
    /// no-slip equality. Equivalent to legged_control's
    /// `centroidalModelType = 0`. Designed to fix the lateral-inversion
    /// and forward-dy cross-coupling that empirical tuning + 12-state
    /// SQP could not eliminate.
    FullCentroidal,
    /// Open-loop "linear-trunk" crawl ([`crate::linear_crawl`]). Holds
    /// the trunk on a strictly +X linear trajectory and serves the legs
    /// around it. Uses [`crate::GaitConfig::four_support_fraction`]
    /// to size the 4-support / 3-support windows. Companion preset is
    /// [`crate::GaitConfig::crawl`] though any preset's
    /// `cycle_period_s` + `swing_height_m` will work; the gait type's
    /// phase offsets are ignored (the linear planner uses its own
    /// diagonal-sequence order).
    LinearCrawl,
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
            GaitMode::CentroidalSrbd => "MPC (centroidal-SRBD)",
            GaitMode::FullCentroidal => "MPC (full-centroidal 24s)",
            GaitMode::LinearCrawl => "Linear crawl (open-loop)",
        }
    }
    pub const ALL: [GaitMode; 5] = [
        GaitMode::Champ,
        GaitMode::Mpc,
        GaitMode::CentroidalSrbd,
        GaitMode::FullCentroidal,
        GaitMode::LinearCrawl,
    ];
}

/// Enum-dispatched wrapper used by `articara` to avoid a `Box<dyn
/// GaitGenerator>` allocation in the per-tick hot path. The variants
/// share the same construction inputs ([`GaitConfig`] +
/// [`KinematicsConfig`]) so switching modes mid-session preserves the
/// model.
pub enum AnyGaitController {
    Champ(ChampGaitController),
    Mpc(crate::MpcGaitController),
    CentroidalSrbd(crate::CentroidalMpcGaitController),
    FullCentroidal(crate::FullCentroidalMpcGaitController),
    LinearCrawl(crate::linear_crawl::LinearCrawlGen),
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
            GaitMode::CentroidalSrbd => AnyGaitController::CentroidalSrbd(
                crate::CentroidalMpcGaitController::new(cfg, kin),
            ),
            GaitMode::FullCentroidal => AnyGaitController::FullCentroidal(
                crate::FullCentroidalMpcGaitController::new(cfg, kin),
            ),
            GaitMode::LinearCrawl => AnyGaitController::LinearCrawl(
                crate::linear_crawl::LinearCrawlGen::new(cfg, kin),
            ),
        }
    }

    pub fn mode(&self) -> GaitMode {
        match self {
            AnyGaitController::Champ(_) => GaitMode::Champ,
            AnyGaitController::Mpc(_) => GaitMode::Mpc,
            AnyGaitController::CentroidalSrbd(_) => GaitMode::CentroidalSrbd,
            AnyGaitController::FullCentroidal(_) => GaitMode::FullCentroidal,
            AnyGaitController::LinearCrawl(_) => GaitMode::LinearCrawl,
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
    /// MPC predicted ground reaction forces from the last tick. Both
    /// `Mpc` (body-root SRBD) and `CentroidalSrbd` (centroidal SRBD)
    /// modes return GRFs via the same `MpcSolution` shape — the
    /// numeric values reflect each mode's underlying model. CHAMP
    /// always returns `None`.
    pub fn predicted_grfs(&self) -> Option<&crate::srbd_mpc::MpcSolution> {
        match self {
            AnyGaitController::Champ(_) => None,
            AnyGaitController::Mpc(c) => c.predicted_grfs(),
            AnyGaitController::CentroidalSrbd(c) => c.predicted_grfs(),
            AnyGaitController::FullCentroidal(c) => c.predicted_grfs(),
            // Open-loop kinematic — no MPC, no GRF prediction.
            AnyGaitController::LinearCrawl(_) => None,
        }
    }

    /// Native centroidal MPC solution, only available in
    /// [`GaitMode::CentroidalSrbd`]. Hosts that want the centroidal-
    /// aware WBC integration read this directly; everyone else can
    /// stick with [`Self::predicted_grfs`] which works in any MPC mode.
    pub fn predicted_centroidal_solution(
        &self,
    ) -> Option<&crate::centroidal_mpc::CentroidalMpcSolution> {
        match self {
            AnyGaitController::CentroidalSrbd(c) => c.predicted_centroidal_solution(),
            _ => None,
        }
    }

    /// Native full-centroidal MPC solution (only in
    /// [`GaitMode::FullCentroidal`]). Hosts that want 24-state-aware
    /// WBC integration read this; everyone else can use
    /// [`Self::predicted_grfs`].
    pub fn predicted_full_centroidal_solution(
        &self,
    ) -> Option<&crate::full_centroidal_mpc::FullCentroidalMpcSolution> {
        match self {
            AnyGaitController::FullCentroidal(c) => c.predicted_full_centroidal_solution(),
            _ => None,
        }
    }

    /// Per-leg stance-foot torque feedforward (`τ = -J^T·f_GRF`)
    /// computed from the last MPC solve.
    pub fn stance_grf_torques(
        &self,
        output: &ControllerOutput,
    ) -> [Option<[f64; 3]>; 4] {
        match self {
            AnyGaitController::Champ(_) => [None; 4],
            AnyGaitController::Mpc(c) => c.stance_grf_torques(output),
            AnyGaitController::CentroidalSrbd(c) => c.stance_grf_torques(output),
            AnyGaitController::FullCentroidal(c) => c.stance_grf_torques(output),
            // Open-loop kinematic — no GRF / no τ_ff.
            AnyGaitController::LinearCrawl(_) => [None; 4],
        }
    }

    /// Override the SRBD MPC's body-mass / inertia / weight matrices.
    /// Only takes effect in [`GaitMode::Mpc`] (body-root SRBD); CHAMP
    /// has no MPC state, and `CentroidalSrbd` uses
    /// [`Self::set_centroidal_mpc_config`] instead.
    pub fn set_srbd_mpc_config(&mut self, cfg: crate::srbd_mpc::SrbdMpcConfig) {
        if let AnyGaitController::Mpc(c) = self {
            c.set_srbd_mpc_config(cfg);
        }
    }

    /// Override the MPC capture-point feedback gain. Applies to all
    /// closed-loop modes (Mpc / CentroidalSrbd / FullCentroidal); CHAMP
    /// has no closed-loop footstep correction and is left untouched.
    /// Default is
    /// [`crate::mpc_controller::DEFAULT_CAPTURE_POINT_GAIN_S`]; pass
    /// `0.0` to disable the closed-loop footstep correction.
    ///
    /// **Stiff-PD note (2026-05-13)**: under `kp ≥ 100 / kv ≤ 1.2` the
    /// `+k·(v_obs - v_cmd)` feedback acts as a positive loop in the
    /// y-axis under pure forward / pure lateral commands — it
    /// amplifies tracking noise instead of damping it (the bug was
    /// masked by URDF-tested soft PD). Until the formula is rewritten
    /// for the linear stance-line model, dynamics-fidelity tests
    /// should call this with `0.0`.
    pub fn set_capture_point_gain(&mut self, k: f64) {
        match self {
            AnyGaitController::Champ(_) => {}
            AnyGaitController::Mpc(c) => c.set_capture_point_gain(k),
            AnyGaitController::CentroidalSrbd(c) => c.set_capture_point_gain(k),
            AnyGaitController::FullCentroidal(c) => c.set_capture_point_gain(k),
            // No closed-loop footstep correction in open-loop crawl.
            AnyGaitController::LinearCrawl(_) => {}
        }
    }

    /// Enable/disable solving the MPC QP on a background thread. Only the
    /// MPC-family controllers have a solver; the open-loop modes ignore
    /// it. The `articara` GUI turns this on so a slow solve can't freeze
    /// the update loop (see [`crate::async_solver`]).
    pub fn set_async_mpc(&mut self, enabled: bool) {
        match self {
            AnyGaitController::Champ(_) => {}
            AnyGaitController::Mpc(c) => c.set_async_mpc(enabled),
            AnyGaitController::CentroidalSrbd(c) => c.set_async_mpc(enabled),
            AnyGaitController::FullCentroidal(c) => c.set_async_mpc(enabled),
            AnyGaitController::LinearCrawl(_) => {}
        }
    }

    /// Set the trunk stance height (m) — the height the body is held at
    /// above the feet. Only [`GaitMode::LinearCrawl`] holds an explicit
    /// trunk-height target; the other modes derive their stance from the
    /// kinematics / SRBD state and ignore this. Clamped to a sane minimum
    /// by the underlying controller. No-op when unset (the LinearCrawl
    /// default is the auto-detected nominal foot height).
    pub fn set_body_height_m(&mut self, h: f64) {
        match self {
            AnyGaitController::LinearCrawl(g) => g.set_body_height_m(h),
            _ => {}
        }
    }

    /// Configure the FullCentroidal controller's nonlinear pulse
    /// branch of the capture-point feedback (η-2 experiment, see
    /// [`crate::mpc_controller::capture_point_step`]). No-op for any
    /// other gait mode — only `FullCentroidal` carries the pulse
    /// fields today; the other closed-loop modes still use pure
    /// linear `k · v_err`.
    pub fn set_capture_point_pulse(&mut self, k_pulse: f64, v_db: f64) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.set_capture_point_pulse(k_pulse, v_db);
        }
    }

    pub fn capture_point_pulse(&self) -> Option<(f64, f64)> {
        match self {
            AnyGaitController::FullCentroidal(c) => Some(c.capture_point_pulse()),
            _ => None,
        }
    }

    /// Activate **goal-pose mode** on the FullCentroidal controller.
    /// `set_velocity_cmd` implicitly clears it. No-op for other gait
    /// modes today — only `FullCentroidal` carries the goal-pose
    /// state.
    pub fn set_goal_pose_world(
        &mut self,
        goal: crate::full_centroidal_controller::GoalPoseWorld,
    ) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.set_goal_pose_world(goal);
        }
    }
    pub fn clear_goal_pose(&mut self) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.clear_goal_pose();
        }
    }
    pub fn goal_pose_world(&self) -> Option<crate::full_centroidal_controller::GoalPoseWorld> {
        match self {
            AnyGaitController::FullCentroidal(c) => c.goal_pose_world(),
            _ => None,
        }
    }

    /// Toggle the FullCentroidal controller's "use MPC-predicted base
    /// for footstep target" path (legged_control-style swing planner
    /// against MPC's optimized body trajectory). No-op for other
    /// modes — only FullCentroidal carries an MPC prediction the
    /// footstep planner can read.
    pub fn set_use_mpc_predicted_footstep(&mut self, enable: bool) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.set_use_mpc_predicted_footstep(enable);
        }
    }
    pub fn use_mpc_predicted_footstep(&self) -> Option<bool> {
        match self {
            AnyGaitController::FullCentroidal(c) => Some(c.use_mpc_predicted_footstep()),
            _ => None,
        }
    }

    /// Read back the active SRBD MPC config. Returns `None` for CHAMP
    /// or `CentroidalSrbd` (use [`Self::centroidal_mpc_config`] for
    /// the centroidal variant).
    pub fn srbd_mpc_config(&self) -> Option<&crate::srbd_mpc::SrbdMpcConfig> {
        match self {
            AnyGaitController::Mpc(c) => Some(c.srbd_mpc_config()),
            _ => None,
        }
    }

    /// Override the centroidal-SRBD MPC's mass / inertia / CoM offset
    /// / weights. Only takes effect in [`GaitMode::CentroidalSrbd`].
    pub fn set_centroidal_mpc_config(
        &mut self,
        cfg: crate::centroidal_mpc::CentroidalMpcConfig,
    ) {
        if let AnyGaitController::CentroidalSrbd(c) = self {
            c.set_centroidal_mpc_config(cfg);
        }
    }

    /// Read the active centroidal MPC config. Returns `None` outside
    /// of [`GaitMode::CentroidalSrbd`].
    pub fn centroidal_mpc_config(
        &self,
    ) -> Option<&crate::centroidal_mpc::CentroidalMpcConfig> {
        match self {
            AnyGaitController::CentroidalSrbd(c) => Some(c.centroidal_mpc_config()),
            _ => None,
        }
    }

    /// Override the full-centroidal MPC's config. No-op outside of
    /// [`GaitMode::FullCentroidal`].
    pub fn set_full_centroidal_mpc_config(
        &mut self,
        cfg: crate::full_centroidal_mpc::FullCentroidalMpcConfig,
    ) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.set_full_centroidal_mpc_config(cfg);
        }
    }

    /// Read the active full-centroidal MPC config.
    pub fn full_centroidal_mpc_config(
        &self,
    ) -> Option<&crate::full_centroidal_mpc::FullCentroidalMpcConfig> {
        match self {
            AnyGaitController::FullCentroidal(c) => Some(c.full_centroidal_mpc_config()),
            _ => None,
        }
    }

    /// Toggle the FullCentroidal controller's legged_control-parity
    /// path (per-step phase prediction + swing-leg vertical foot
    /// velocity equality constraint). No-op outside of
    /// [`GaitMode::FullCentroidal`].
    pub fn set_legged_control_parity(&mut self, enable: bool) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.set_legged_control_parity(enable);
        }
    }

    /// Read whether the FullCentroidal controller's parity path is
    /// active. Returns `None` outside of [`GaitMode::FullCentroidal`].
    pub fn legged_control_parity(&self) -> Option<bool> {
        match self {
            AnyGaitController::FullCentroidal(c) => Some(c.legged_control_parity()),
            _ => None,
        }
    }

    /// Toggle whether the parity path uses the URDF nominal pose as
    /// the joint_q tracking reference (matches legged_control's
    /// `DEFAULT_JOINT_STATE`). No-op outside FullCentroidal mode.
    pub fn set_parity_use_nominal_q_ref(&mut self, enable: bool) {
        if let AnyGaitController::FullCentroidal(c) = self {
            c.set_parity_use_nominal_q_ref(enable);
        }
    }

    pub fn parity_use_nominal_q_ref(&self) -> Option<bool> {
        match self {
            AnyGaitController::FullCentroidal(c) => Some(c.parity_use_nominal_q_ref()),
            _ => None,
        }
    }
}

impl GaitGenerator for AnyGaitController {
    fn tick(&mut self, dt: f64) -> ControllerOutput {
        match self {
            AnyGaitController::Champ(c) => c.tick(dt),
            AnyGaitController::Mpc(c) => c.tick(dt),
            AnyGaitController::CentroidalSrbd(c) => c.tick(dt),
            AnyGaitController::FullCentroidal(c) => c.tick(dt),

            AnyGaitController::LinearCrawl(c) => c.tick(dt),
        }
    }
    fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        match self {
            AnyGaitController::Champ(c) => c.set_velocity_cmd(cmd),
            AnyGaitController::Mpc(c) => c.set_velocity_cmd(cmd),
            AnyGaitController::CentroidalSrbd(c) => c.set_velocity_cmd(cmd),
            AnyGaitController::FullCentroidal(c) => c.set_velocity_cmd(cmd),

            AnyGaitController::LinearCrawl(c) => c.set_velocity_cmd(cmd),
        }
    }
    fn velocity_cmd(&self) -> VelocityCmd {
        match self {
            AnyGaitController::Champ(c) => c.velocity_cmd(),
            AnyGaitController::Mpc(c) => c.velocity_cmd(),
            AnyGaitController::CentroidalSrbd(c) => c.velocity_cmd(),
            AnyGaitController::FullCentroidal(c) => c.velocity_cmd(),

            AnyGaitController::LinearCrawl(c) => c.velocity_cmd(),
        }
    }
    fn reset(&mut self) {
        match self {
            AnyGaitController::Champ(c) => c.reset(),
            AnyGaitController::Mpc(c) => c.reset(),
            AnyGaitController::CentroidalSrbd(c) => c.reset(),
            AnyGaitController::FullCentroidal(c) => c.reset(),

            AnyGaitController::LinearCrawl(c) => c.reset(),
        }
    }
    fn config(&self) -> &GaitConfig {
        match self {
            AnyGaitController::Champ(c) => c.config(),
            AnyGaitController::Mpc(c) => c.config(),
            AnyGaitController::CentroidalSrbd(c) => c.config(),
            AnyGaitController::FullCentroidal(c) => c.config(),

            AnyGaitController::LinearCrawl(c) => c.config(),
        }
    }
    fn set_config(&mut self, cfg: GaitConfig) {
        match self {
            AnyGaitController::Champ(c) => c.set_config(cfg),
            AnyGaitController::Mpc(c) => c.set_config(cfg),
            AnyGaitController::CentroidalSrbd(c) => c.set_config(cfg),
            AnyGaitController::FullCentroidal(c) => c.set_config(cfg),

            AnyGaitController::LinearCrawl(c) => c.set_config(cfg),
        }
    }
    fn kinematics(&self) -> &KinematicsConfig {
        match self {
            AnyGaitController::Champ(c) => c.kinematics(),
            AnyGaitController::Mpc(c) => c.kinematics(),
            AnyGaitController::CentroidalSrbd(c) => c.kinematics(),
            AnyGaitController::FullCentroidal(c) => c.kinematics(),

            AnyGaitController::LinearCrawl(c) => c.kinematics(),
        }
    }
    fn set_kinematics(&mut self, kin: KinematicsConfig) {
        match self {
            AnyGaitController::Champ(c) => c.set_kinematics(kin),
            AnyGaitController::Mpc(c) => c.set_kinematics(kin),
            AnyGaitController::CentroidalSrbd(c) => c.set_kinematics(kin),
            AnyGaitController::FullCentroidal(c) => c.set_kinematics(kin),

            AnyGaitController::LinearCrawl(c) => c.set_kinematics(kin),
        }
    }
    fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        match self {
            AnyGaitController::Champ(c) => c.set_knee_forward(leg, forward),
            AnyGaitController::Mpc(c) => c.set_knee_forward(leg, forward),
            AnyGaitController::CentroidalSrbd(c) => c.set_knee_forward(leg, forward),
            AnyGaitController::FullCentroidal(c) => c.set_knee_forward(leg, forward),

            AnyGaitController::LinearCrawl(c) => c.set_knee_forward(leg, forward),
        }
    }
    fn set_knee_pattern(&mut self, pattern: KneePattern) {
        match self {
            AnyGaitController::Champ(c) => c.set_knee_pattern(pattern),
            AnyGaitController::Mpc(c) => c.set_knee_pattern(pattern),
            AnyGaitController::CentroidalSrbd(c) => c.set_knee_pattern(pattern),
            AnyGaitController::FullCentroidal(c) => c.set_knee_pattern(pattern),

            AnyGaitController::LinearCrawl(c) => c.set_knee_pattern(pattern),
        }
    }
    fn knee_pattern(&self) -> KneePattern {
        match self {
            AnyGaitController::Champ(c) => c.knee_pattern(),
            AnyGaitController::Mpc(c) => c.knee_pattern(),
            AnyGaitController::CentroidalSrbd(c) => c.knee_pattern(),
            AnyGaitController::FullCentroidal(c) => c.knee_pattern(),

            AnyGaitController::LinearCrawl(c) => c.knee_pattern(),
        }
    }
    fn knee_forward(&self) -> [bool; 4] {
        match self {
            AnyGaitController::Champ(c) => c.knee_forward(),
            AnyGaitController::Mpc(c) => c.knee_forward(),
            AnyGaitController::CentroidalSrbd(c) => c.knee_forward(),
            AnyGaitController::FullCentroidal(c) => c.knee_forward(),

            AnyGaitController::LinearCrawl(c) => c.knee_forward(),
        }
    }
    fn set_body_state_observed(
        &mut self,
        v_world: Vector3<f64>,
        omega_world: Vector3<f64>,
    ) {
        match self {
            AnyGaitController::Champ(c) => c.set_body_state_observed(v_world, omega_world),
            AnyGaitController::Mpc(c) => c.set_body_state_observed(v_world, omega_world),
            AnyGaitController::CentroidalSrbd(c) => {
                c.set_body_state_observed(v_world, omega_world)
            }
            AnyGaitController::FullCentroidal(c) => {
                c.set_body_state_observed(v_world, omega_world)
            }

            AnyGaitController::LinearCrawl(c) => {
                c.set_body_state_observed(v_world, omega_world)
            }
        }
    }
    fn set_body_pose_observed(
        &mut self,
        world_yaw: f64,
        world_position: Vector3<f64>,
    ) {
        match self {
            // CHAMP doesn't reference body_state externally — keep its
            // diagnostics in sync anyway so the panel reads the real
            // pose if the host queries it later.
            AnyGaitController::Champ(_) => {}
            AnyGaitController::Mpc(c) => c.set_body_pose_observed(world_yaw, world_position),
            AnyGaitController::CentroidalSrbd(c) => {
                c.set_body_pose_observed(world_yaw, world_position)
            }
            AnyGaitController::FullCentroidal(c) => {
                c.set_body_pose_observed(world_yaw, world_position)
            }

            AnyGaitController::LinearCrawl(c) => {
                c.set_body_pose_observed(world_yaw, world_position)
            }
        }
    }
}
