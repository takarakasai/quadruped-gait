//! 24-state full-centroidal gait controller (D3.3.5).
//!
//! Architectural sibling of [`crate::CentroidalMpcGaitController`]:
//! identical open-loop layer (CHAMP-style phase + Raibert footstep +
//! analytical 3R IK → 12 joint targets), but swaps in
//! [`crate::FullCentroidalMpc`] for the closed-loop GRF + joint-velocity
//! prediction.
//!
//! What changes vs. the 12-state centroidal controller:
//!
//! 1. The MPC state now carries the 12 leg joint angles, so the
//!    per-node moment arm `r = R · (foot_body(q) − com_offset)` updates
//!    when the optimiser perturbs `joint_q` within the horizon. The
//!    body-root SRBD and the centroidal-SRBD couldn't see this coupling.
//! 2. The MPC equality constraint set includes **stance no-slip**
//!    (`v_foot_world = 0` per stance-leg-step), expressed linearly in
//!    the condensed QP via the lifted state. This forces the solution
//!    `joint_v` to be physically consistent with a pinned foot.
//! 3. Reference joint_q is held at the controller's current IK output
//!    (D3.3.5a — design choice (a) from the planning session). Swing
//!    leg foot tracking is still driven by the CHAMP layer's joint
//!    target.
//!
//! ## D3.3.5b — legged_control parity (opt-in)
//!
//! When [`FullCentroidalMpcGaitController::set_legged_control_parity`]
//! is `true`, two additional behaviours kick in to match OCS2 /
//! legged_control's `centroidalModelType = 0` setup:
//!
//! - The per-step contact schedule is built from a per-leg phase
//!   projection (`cycle_phase + k·dt_per_step / cycle_period_s +
//!   offset_leg mod 1`), rather than the D3.3.5a `duty > 0.5 ? all
//!   stance : all swing` proxy.
//! - Each swing-leg-step receives a planned world-frame vertical foot
//!   velocity from [`crate::swing_traj::swing_vz_world`], which the
//!   MPC's new `enable_swing_normal_velocity_constraint` equality
//!   tracks per node (mirrors `NormalVelocityConstraintCppAd`).
//!
//! Joint_q reference is **still held constant** under parity, exactly
//! as legged_control does — the swing arc enters the MPC in task
//! space, not joint space. The legacy path (parity off) remains the
//! default and is the basis of the existing benchmark rows; the
//! parity path is exposed for A/B comparison.
//!
//! GRF output is projected into [`MpcSolution`] via
//! [`to_compat_mpc_solution_full`] so WBC integration stays
//! mode-agnostic. The native solution is available via
//! [`Self::predicted_full_centroidal_solution`].

use nalgebra::{Matrix3, Vector3};

use crate::async_solver::AsyncJobWorker;
use crate::body_state::BodyState;
use crate::bound_reference::BoundTrimConfig;
use crate::config::{GaitConfig, GaitType, KinematicsConfig, LegId, LegKinematics, VelocityCmd};
use crate::controller::{ControllerOutput, LegOutput};
use crate::footstep::Footstep;
use crate::full_centroidal_mpc::{
    FullCentroidalContactSchedule, FullCentroidalInput, FullCentroidalMpc,
    FullCentroidalMpcConfig, FullCentroidalMpcSolution, FullCentroidalReference,
    FullCentroidalState, N_FEET, N_LEG_JOINTS,
};
use crate::ik::{foot_jacobian_body, forward_leg_kinematics, solve_leg_ik, LegIkSolution};
use crate::mpc_controller::{
    body_to_world_horizontal, effective_swing_height, make_leg_output,
    world_to_body_horizontal, DEFAULT_CAPTURE_POINT_GAIN_S, HORIZON_STEPS,
    MIN_HALF_FRACTION, STANCE_GRF_MIN_N,
};
use crate::phase::PhaseGenerator;
use crate::srbd_mpc::{MpcSolution, SrbdState};
use crate::swing_traj::{swing_position, swing_vz_world};

#[derive(Clone, Debug)]
pub struct FullCentroidalMpcGaitController {
    cfg: GaitConfig,
    kin: KinematicsConfig,
    phase_gen: PhaseGenerator,
    body_state: BodyState,
    cmd: VelocityCmd,
    knee_forward: [bool; 4],

    k_capture: f64,
    /// Pulse-branch slope past `v_capture_deadband` (see
    /// [`crate::mpc_controller::capture_point_step`]). 0 disables the
    /// nonlinear pulse; the controller then falls back to a pure
    /// linear `k_capture · v_err` response. Defaults to 0.
    k_capture_pulse: f64,
    /// Deadband (m/s) below which the pulse branch contributes
    /// nothing. Defaults to 0 — i.e. the pulse acts on all v_err
    /// magnitudes when `k_capture_pulse > 0`. Tuned to ≈ 0.05 m/s in
    /// the η-2 experiment so cycle-noise on `v_err_y` doesn't trigger
    /// a foothold shift while real pushes (> 0.05 m/s = ~ 4 N impulse)
    /// still get the steeper response.
    v_capture_deadband: f64,
    /// Bound-specific fore-aft (body-x) foot-placement feedback gain
    /// (seconds), applied ON TOP of the cmd-based Raibert `half` and
    /// INDEPENDENT of [`Self::k_capture`]. Adds `k · v_err_body.x` to
    /// the fore-aft half-step — the classic Raibert running/bounding
    /// speed regulator (`x_foot = v̄·T_st/2 + k·(v−v_des)`), where
    /// during the flight phase the landing spot is the only authority
    /// over fore-aft speed. Deliberately x-ONLY: re-enabling the
    /// generic (x+y) `k_capture` for Bound reacted to lateral velocity
    /// NOISE and induced a roll instability (`articara/ref/
    /// wbc_comparison.md` Sec.5bt), whereas the fore-aft error is the
    /// real speed-tracking signal Bound needs to close. `0.0` (default)
    /// preserves prior behaviour exactly. See Sec.5c6/5c7 (local doc).
    bound_fore_aft_placement_gain: f64,
    /// Low-pass (EMA) estimate of the measured body-frame fore-aft
    /// velocity, used as the Raibert **neutral-point** speed when
    /// [`Self::bound_fore_aft_placement_gain`] is active (Sec.5c7,
    /// local doc). The neutral foothold `ẋ·T_st/2` must be sized by the
    /// speed the robot is ACTUALLY going, not the (possibly
    /// unreachable) commanded speed -- otherwise, when the command
    /// exceeds what the gait can hold, the cmd-based neutral over-places
    /// the foot and the feedback biases chronically (Sec.5c6's negative
    /// result). Filtered (not raw `v_obs`) to reject per-tick observer
    /// noise. Updated every [`Self::tick`]; never read when the gain
    /// is 0, so it costs nothing on the default path.
    v_fore_aft_filtered: f64,
    v_observed_world: Vector3<f64>,
    omega_observed_world: Vector3<f64>,
    /// Observed base attitude (world-frame roll/pitch), fed by
    /// [`Self::set_body_attitude_observed`]. Only consumed by the
    /// Poincaré/deadbeat pitch foot-placement (Sec.5f6); defaults to
    /// zero so callers that never set it keep the prior behaviour.
    roll_observed: f64,
    pitch_observed: f64,
    /// **Poincaré/deadbeat pitch foot-placement** (Sec.5f6). For an
    /// energetic Bound with real air time, the tumble is a slow pitch
    /// divergence: each cycle imparts a small net pitch impulse the
    /// stance can't cancel, so pitch angular momentum grows until the
    /// body somersaults (Sec.5f3-5f5: rate-deadbeat state weights only
    /// delay it 4x). The touchdown fore-aft position of the front/rear
    /// pair sets the pitch moment during the next stance (upward GRF at
    /// fore-aft offset x → pitch moment ∝ -x), so shifting the foothold
    /// by the pitch error is a Poincaré-section deadbeat that nulls the
    /// accumulated momentum rather than merely damping it:
    ///   `x_foot += k_ang·pitch + k_rate·pitch_rate`.
    /// Applied uniformly to both pairs (same body pitch state); the sign
    /// is carried in the gains and swept empirically (Sec.5f's pitch sign
    /// convention is antiphase to `euler_angles()`, see the trim path).
    /// `(0.0, 0.0)` (default) leaves the foothold untouched.
    bound_pitch_placement_gain: f64,
    bound_pitch_rate_placement_gain: f64,
    /// Sec.5f8 DC-blocker time constant (s) for the pitch foot-placement.
    /// The orbit-relative deadbeat still leaves a persistent forward
    /// foothold bias (the trim closed-form nominal ≠ the real orbit's
    /// pitch_rate at the sampled phase), which drags the body backward.
    /// A slow EMA of the applied shift estimates that DC bias; subtracting
    /// it leaves only the AC (deviation-stabilizing) part, so steady-state
    /// placement is drift-neutral while transient corrections survive.
    /// `0.0` (default) disables it -- the raw Sec.5f6/5f8 shift.
    bound_pitch_placement_dc_tau: f64,
    /// Sec.5f9 (P2) tabulated forward-Bound reference orbit: rows of
    /// `[phase, z, pitch, vx, vz, w]` over one cycle (phase in [0,1)),
    /// produced offline by the trajopt existence solver
    /// (`ref/scripts/bound_trajopt_p0_shooting.py`). When set, the MPC
    /// reference loop injects the phase-interpolated (z, pitch, vx, vz, w)
    /// instead of the flat / closed-form-trim reference -- giving the MPC
    /// a CONSISTENT feasible forward orbit to track (the missing piece
    /// §5f7/5f8 identified: forward velocity + trim pitch + flat height
    /// were mutually infeasible, so vx collapsed to ~0). Rows must be
    /// sorted by ascending phase. `None` keeps the existing reference.
    bound_tabulated_reference: Option<Vec<[f64; 6]>>,
    /// Running EMA of the applied pitch shift (the DC estimate) and the
    /// once-per-tick DC-blocked value the footstep planner consumes. The
    /// shift is leg-independent (global pitch state), so it is computed
    /// once per `tick` (where `&mut self` allows the EMA update) and read
    /// by every leg's `compute_mpc_footstep`.
    pitch_placement_shift_dc: f64,
    pitch_placement_shift: f64,

    full_centroidal_mpc: FullCentroidalMpc,
    last_solution: Option<FullCentroidalMpcSolution>,
    last_solution_compat: Option<MpcSolution>,
    mpc_solve_accumulator_s: f64,

    /// **A1 lock-at-swing-entry**: per-leg cache of the body-frame
    /// touch_down captured at the moment a swing phase begins. While
    /// `mpc_optimized_footstep` is on, the per-tick swing IK target
    /// reads this cache instead of recomputing from the latest MPC
    /// prediction every 2 ms — otherwise mid-swing wobble (the MPC
    /// solves on a 30 ms cadence and its predicted joint_q at
    /// step k_td shifts each solve, while sub_fraction's tick-rate
    /// advance also re-aims k_td) drives oscillation that the v3
    /// bench showed amounts to a full body topple. Cleared at
    /// touchdown so the next swing captures a fresh value.
    swing_locked_touch_down_body: [Option<Vector3<f64>>; N_FEET],
    /// **A1 lock state**: previous tick's `is_stance` per leg, used
    /// only to detect the stance→swing transition that triggers
    /// the [`Self::swing_locked_touch_down_body`] capture.
    prev_leg_is_stance: [bool; N_FEET],

    /// When `true`, the MPC's contact schedule is built from a per-leg
    /// per-step phase projection (matching legged_control's
    /// `SwitchedModelReferenceManager` behaviour), and each swing-leg-
    /// step receives a planned vertical foot velocity that the MPC
    /// enforces via the `NormalVelocityConstraintCppAd`-equivalent
    /// equality (see
    /// [`FullCentroidalMpcConfig::enable_swing_normal_velocity_constraint`]).
    ///
    /// Default `false` — the legacy D3.3.5a path stays available for
    /// A/B comparison via the external-force robustness benchmark and
    /// the Rhai test scripts.
    legged_control_parity: bool,

    /// When `true` AND [`Self::legged_control_parity`] is also `true`,
    /// the joint_q tracking reference is filled with the URDF nominal
    /// stance pose (= per-leg analytical IK of
    /// `kin.nominal_foot_body`) instead of the observed `joint_q_now`.
    /// This matches legged_control's
    /// `DEFAULT_JOINT_STATE`-based tracking (see `reference.info`),
    /// where the MPC's joint cost biases the swing leg back toward the
    /// nominal pose rather than tracking whatever the leg is doing
    /// right now. Independent of [`Self::legged_control_parity`] so the
    /// β-only variant (parity ON, nominal_q_ref OFF) and the combined
    /// (α+β) variant can both be benchmarked.
    parity_use_nominal_q_ref: bool,

    /// Optional absolute goal pose (world frame). When `Some`, [`Self::tick`]
    /// recomputes the velocity command from `goal − observed_pose` at each
    /// tick, so the body actively tracks back toward the goal after a
    /// disturbance — mirroring legged_control's
    /// `goalToTargetTrajectories` path. When `None`, the controller
    /// uses [`Self::cmd`] verbatim (= legged_control's
    /// `cmdVelToTargetTrajectories` path).
    goal_pose: Option<GoalPoseWorld>,

    /// When `true`, [`Self::compute_mpc_footstep`] adds a body-frame
    /// correction term derived from the **MPC's predicted base
    /// position** at one swing-duration ahead. This makes foot
    /// placement track the body trajectory the MPC actually plans
    /// (which already accounts for disturbances via the MPC's GRF +
    /// state-cost loop), rather than the open-loop cmd + linear
    /// capture-point heuristic — the same architectural pattern that
    /// legged_control's `SwingTrajectoryPlanner` uses against OCS2's
    /// predicted base trajectory. The capture-point feedback term is
    /// suppressed internally while this flag is on so the two paths
    /// don't double-correct.
    ///
    /// Default `false` for backward compatibility. Requires a
    /// previously-solved MPC solution to read; if `last_solution` is
    /// `None` the flag silently falls back to the cap-pt path.
    use_mpc_predicted_footstep: bool,

    /// When `true`, the joint_q tracking reference fed to the MPC at
    /// each horizon step `k` is no longer a flat hold — it's sampled
    /// from the same open-loop swing/stance foot curve `tick()` already
    /// uses (`Footstep::stance_at` / `swing_position`, IK-inverted via
    /// `solve_leg_ik`) at that step's *projected* phase, for every leg,
    /// stance or swing. This is the D3.3.5a simplification's reversal:
    /// instead of the MPC's joint-space cost being indifferent to what
    /// the swing leg is actually planned to do over the horizon, it now
    /// tracks the real planned arc — closer to legged_control's
    /// whole-body joint reference, without touching the MPC's own cost
    /// weights or dynamics.
    ///
    /// Requires `legged_control_parity` for the projected per-step
    /// phase (`legacy` mode's crude "duty>0.5 ⇒ all stance" proxy
    /// doesn't carry per-leg sub-fraction info past step 0, so this
    /// flag is a no-op without parity). Takes priority over
    /// `parity_use_nominal_q_ref` when both are set. Default `false`.
    dynamic_joint_q_reference: bool,

    /// When `true` (and `self.cfg.gait_type == GaitType::Bound`), the
    /// per-horizon-step reference built by
    /// [`Self::build_full_centroidal_inputs`] is augmented with the
    /// closed-form Bound "trim" pitch / fore-aft-GRF profile
    /// ([`crate::BoundTrimConfig`]) instead of the flat zero-pitch,
    /// zero-`F_x` hold every gait has used until now (`grfs[leg].x`
    /// was never set anywhere in this function before this flag was
    /// added — see `articara/ref/wbc_comparison.md` Sec.5bb/5bc for
    /// the derivation and the first-principles feasibility check that
    /// motivated it). A no-op for every other gait, and a no-op for
    /// Bound too while the velocity command is zero (holding). Default
    /// `false`.
    enable_bound_trim_reference: bool,

    /// Fraction (`[0,1]`) of the friction-clipped trim force actually
    /// commanded when [`Self::enable_bound_trim_reference`] is on
    /// (`BoundTrimConfig::thrust_scale`). `1.0` (default) reproduces
    /// the original behaviour -- and at Go2's real numbers, already
    /// saturates the hard friction cone by itself, leaving zero
    /// headroom for this same MPC's own velocity-tracking `F_x` (see
    /// `articara/ref/wbc_comparison.md` Sec.5bf). Values `<1.0`
    /// deliberately under-cancel pitch torque to free real friction
    /// budget for velocity tracking, at the cost of a larger
    /// `theta_peak`.
    bound_trim_thrust_scale: f64,

    /// If `Some(fraction)`, `BoundTrimConfig::velocity_ripple_fraction`
    /// -- sizes the trim's `F_x` from a target velocity ripple
    /// (fraction of `self.cmd.vx`) instead of from
    /// `bound_trim_thrust_scale`, the MIT-Cheetah-style "impulse
    /// scaling" alternative (`articara/ref/wbc_comparison.md` Sec.5bj).
    /// `None` (default) preserves the `bound_trim_thrust_scale`-based
    /// behaviour exactly.
    bound_trim_velocity_ripple_fraction: Option<f64>,

    /// Background worker that runs the full-centroidal SQP off the
    /// caller's thread. This is the heaviest MPC (≈0.4 s/solve), so
    /// solving it inline on the GUI's update loop froze the window —
    /// the whole reason this async path exists. See
    /// [`crate::async_solver`] and [`Self::set_async_mpc`].
    mpc_worker: AsyncJobWorker<FullCentroidalMpcSolution>,
    /// When `true`, solve on [`Self::mpc_worker`] instead of inline.
    /// Default `false` (synchronous, deterministic).
    async_mpc: bool,
}

/// Absolute target pose in the world frame, with traverse-speed limits.
/// See [`FullCentroidalMpcGaitController::set_goal_pose_world`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalPoseWorld {
    /// Target world-frame x position (m).
    pub x_m: f64,
    /// Target world-frame y position (m).
    pub y_m: f64,
    /// Target world-frame yaw (rad), wrapped to (−π, π] when used.
    pub yaw_rad: f64,
    /// Maximum linear traverse speed (m/s) the controller is allowed to
    /// command toward the goal. The instantaneous velocity command is
    /// `clamp(distance_to_goal / time_to_goal, ±max_v_m_s)`.
    pub max_v_m_s: f64,
    /// Maximum yaw rate (rad/s) the controller is allowed to command.
    pub max_wz_rad_s: f64,
    /// Position tolerance: when the body is within this radius of the
    /// goal (xy) AND `|yaw_err| < yaw_tolerance_rad`, the controller
    /// issues `VelocityCmd::zero()` so the gait holds in stance.
    pub position_tolerance_m: f64,
    pub yaw_tolerance_rad: f64,
}

impl FullCentroidalMpcGaitController {
    pub fn new(cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        let phase_gen = PhaseGenerator::new(cfg.clone());
        // Default config uses a placeholder KinematicsConfig (Cheetah-3
        // class). The host's auto_detect overrides it via
        // `set_full_centroidal_mpc_config` at `GaitController::build`
        // time, slotting in the per-leg analytical FK params for this
        // specific URDF.
        let mut mpc_cfg = FullCentroidalMpcConfig::default_with_kin(kin.clone());
        let _ = &mut mpc_cfg;
        Self {
            cfg,
            kin,
            phase_gen,
            body_state: BodyState::new(),
            cmd: VelocityCmd::zero(),
            knee_forward: [false; 4],
            k_capture: DEFAULT_CAPTURE_POINT_GAIN_S,
            k_capture_pulse: 0.0,
            v_capture_deadband: 0.0,
            bound_fore_aft_placement_gain: 0.0,
            roll_observed: 0.0,
            pitch_observed: 0.0,
            bound_pitch_placement_gain: 0.0,
            bound_pitch_rate_placement_gain: 0.0,
            bound_pitch_placement_dc_tau: 0.0,
            pitch_placement_shift_dc: 0.0,
            pitch_placement_shift: 0.0,
            bound_tabulated_reference: None,
            v_fore_aft_filtered: 0.0,
            v_observed_world: Vector3::zeros(),
            omega_observed_world: Vector3::zeros(),
            full_centroidal_mpc: FullCentroidalMpc::new(mpc_cfg),
            last_solution: None,
            last_solution_compat: None,
            mpc_solve_accumulator_s: f64::INFINITY,
            swing_locked_touch_down_body: [None; N_FEET],
            prev_leg_is_stance: [true; N_FEET],
            legged_control_parity: false,
            parity_use_nominal_q_ref: false,
            goal_pose: None,
            use_mpc_predicted_footstep: false,
            dynamic_joint_q_reference: false,
            enable_bound_trim_reference: false,
            bound_trim_thrust_scale: 1.0,
            bound_trim_velocity_ripple_fraction: None,
            mpc_worker: AsyncJobWorker::new(),
            async_mpc: false,
        }
    }

    /// Enable/disable solving the MPC SQP on a background thread. Off by
    /// default (synchronous). The `articara` GUI enables it so a slow
    /// solve can't stall the eframe update loop and freeze the window.
    pub fn set_async_mpc(&mut self, enabled: bool) {
        self.async_mpc = enabled;
    }

    pub fn use_mpc_predicted_footstep(&self) -> bool {
        self.use_mpc_predicted_footstep
    }
    /// Toggle the "MPC-predicted base → footstep target" path
    /// described on the struct field. When the flag is enabled the
    /// linear cap-pt feedback is dropped from `compute_mpc_footstep`
    /// — both paths target the same disturbance correction and
    /// stacking them double-counts the response. The horizon-bias
    /// term is dropped for the same reason.
    pub fn set_use_mpc_predicted_footstep(&mut self, enable: bool) {
        self.use_mpc_predicted_footstep = enable;
    }

    pub fn legged_control_parity(&self) -> bool {
        self.legged_control_parity
    }
    /// Toggle the legged_control-style swing-leg vertical foot velocity
    /// constraint path. Also flips the MPC config's
    /// `enable_swing_normal_velocity_constraint` to keep the two in
    /// sync — the controller is the only writer of that flag in
    /// practice.
    pub fn set_legged_control_parity(&mut self, enable: bool) {
        self.legged_control_parity = enable;
        let mut mpc_cfg = self.full_centroidal_mpc.config().clone();
        mpc_cfg.enable_swing_normal_velocity_constraint = enable;
        self.full_centroidal_mpc.set_config(mpc_cfg);
    }

    pub fn parity_use_nominal_q_ref(&self) -> bool {
        self.parity_use_nominal_q_ref
    }
    /// Switch the joint_q tracking reference between the observed
    /// `joint_q_now` (default) and the URDF nominal stance pose. Only
    /// takes effect while [`Self::legged_control_parity`] is also on.
    /// See struct docs for the rationale.
    pub fn set_parity_use_nominal_q_ref(&mut self, enable: bool) {
        self.parity_use_nominal_q_ref = enable;
    }

    pub fn dynamic_joint_q_reference(&self) -> bool {
        self.dynamic_joint_q_reference
    }
    /// Toggle the per-horizon-step dynamic joint_q reference. See the
    /// struct field's doc comment — requires `legged_control_parity`
    /// to have an effect.
    pub fn set_dynamic_joint_q_reference(&mut self, enable: bool) {
        self.dynamic_joint_q_reference = enable;
    }

    pub fn task_space_joint_vel_weight(&self) -> Option<[f64; 3]> {
        self.full_centroidal_mpc.config().joint_vel_nominal_jacobian.is_some()
            .then(|| self.full_centroidal_mpc.config().r_taskspace_joint_vel)
    }
    /// Replace the flat per-joint `r_diag[12..24]` joint_v cost with a
    /// task-space (foot-velocity) weight mapped through each leg's
    /// fixed-nominal-pose Jacobian — legged_control/OCS2's own
    /// technique (`LeggedRobotInterface::initializeInputCostWeight` in
    /// `ocs2_legged_robot`, confirmed against `ref/ocs2`). The nominal
    /// pose used is each leg's `kin.nominal_foot_body` at the
    /// controller's current `knee_forward` convention — same pose the
    /// β nominal-`joint_q` path (`parity_use_nominal_q_ref`) already
    /// uses, computed once here rather than cached, since this is only
    /// called on config changes, not per tick.
    ///
    /// Pass `None` to revert to the flat diagonal (default).
    pub fn set_task_space_joint_vel_weight(&mut self, r_taskspace: Option<[f64; 3]>) {
        let mut mpc_cfg = self.full_centroidal_mpc.config().clone();
        match r_taskspace {
            Some(r) => {
                let mut jacobians = [Matrix3::zeros(); N_FEET];
                for slot in 0..N_FEET {
                    let kin = self.kin.leg(LegId::ALL[slot]);
                    let knee_fwd = self.knee_forward[slot];
                    let sol = solve_leg_ik(kin, kin.nominal_foot_body, knee_fwd);
                    let (h, th, c) = sol.angles();
                    jacobians[slot] = foot_jacobian_body(kin, h, th, c);
                }
                mpc_cfg.joint_vel_nominal_jacobian = Some(jacobians);
                mpc_cfg.r_taskspace_joint_vel = r;
            }
            None => {
                mpc_cfg.joint_vel_nominal_jacobian = None;
            }
        }
        self.full_centroidal_mpc.set_config(mpc_cfg);
    }

    pub fn true_centroidal_coupling(&self) -> bool {
        self.full_centroidal_mpc.config().enable_true_centroidal_coupling
    }
    /// Toggle the true-centroidal-coupling bias term (see
    /// [`FullCentroidalMpcConfig`]'s doc comment) — a no-op if the
    /// config's `true_centroidal_coupling_data` wasn't populated (no
    /// `misarta` model was available at auto-detect time).
    pub fn set_true_centroidal_coupling(&mut self, enable: bool) {
        let mut mpc_cfg = self.full_centroidal_mpc.config().clone();
        mpc_cfg.enable_true_centroidal_coupling = enable;
        self.full_centroidal_mpc.set_config(mpc_cfg);
    }

    pub fn bound_trim_reference(&self) -> bool {
        self.enable_bound_trim_reference
    }
    /// Toggle the closed-form Bound trim reference (see
    /// [`Self::enable_bound_trim_reference`]'s doc comment). A no-op
    /// unless `self.cfg.gait_type == GaitType::Bound`.
    pub fn set_bound_trim_reference(&mut self, enable: bool) {
        self.enable_bound_trim_reference = enable;
    }

    pub fn bound_trim_thrust_scale(&self) -> f64 {
        self.bound_trim_thrust_scale
    }
    /// Set the partial-trim fraction (see
    /// [`Self::bound_trim_thrust_scale`]'s doc comment). Clamped to
    /// `[0,1]`; a no-op unless [`Self::set_bound_trim_reference`] is
    /// also enabled.
    pub fn set_bound_trim_thrust_scale(&mut self, thrust_scale: f64) {
        self.bound_trim_thrust_scale = thrust_scale.clamp(0.0, 1.0);
    }

    pub fn bound_trim_velocity_ripple_fraction(&self) -> Option<f64> {
        self.bound_trim_velocity_ripple_fraction
    }
    /// Set the "impulse scaling" velocity-ripple fraction (see
    /// [`Self::bound_trim_velocity_ripple_fraction`]'s doc comment).
    /// `Some(fraction)` takes priority over `bound_trim_thrust_scale`;
    /// `None` reverts to the `thrust_scale`-based path.
    pub fn set_bound_trim_velocity_ripple_fraction(&mut self, fraction: Option<f64>) {
        self.bound_trim_velocity_ripple_fraction = fraction;
    }

    pub fn goal_pose_world(&self) -> Option<GoalPoseWorld> {
        self.goal_pose
    }
    /// Activate **goal-pose mode**: at each [`Self::tick`] the velocity
    /// command is recomputed from `(goal − observed_pose) / t_to_goal`,
    /// rotated into the body frame, and saturated at the configured
    /// `max_v / max_wz`. Equivalent to legged_control's
    /// `goalToTargetTrajectories` path — when the body is pushed off
    /// course, the recomputed cmd has a non-zero component pointing
    /// back at the goal, so the controller actively recovers position.
    ///
    /// Cleared by [`Self::clear_goal_pose`] or by setting a new
    /// velocity command via [`Self::set_velocity_cmd`] (the latter
    /// implicitly disables goal mode so existing callers that only use
    /// the velocity API don't see surprising drift).
    pub fn set_goal_pose_world(&mut self, goal: GoalPoseWorld) {
        self.goal_pose = Some(goal);
    }
    pub fn clear_goal_pose(&mut self) {
        self.goal_pose = None;
    }

    pub fn predicted_grfs(&self) -> Option<&MpcSolution> {
        self.last_solution_compat.as_ref()
    }

    pub fn predicted_full_centroidal_solution(
        &self,
    ) -> Option<&FullCentroidalMpcSolution> {
        self.last_solution.as_ref()
    }

    pub fn stance_grf_torques(
        &self,
        output: &ControllerOutput,
    ) -> [Option<[f64; 3]>; 4] {
        let mut out = [None; 4];
        let Some(sol) = self.last_solution_compat.as_ref() else {
            return out;
        };
        if !sol.solved {
            return out;
        }
        let yaw = self.body_state.world_yaw;
        let (sy, cy) = yaw.sin_cos();
        for slot in 0..N_FEET {
            let leg_out = &output.legs[slot];
            if !leg_out.phase.is_stance {
                continue;
            }
            let f_world = sol.grfs_first_step[slot];
            if f_world.norm() < STANCE_GRF_MIN_N {
                continue;
            }
            let f_body = Vector3::new(
                cy * f_world.x + sy * f_world.y,
                -sy * f_world.x + cy * f_world.y,
                f_world.z,
            );
            let kin_leg = self.kin.leg(LegId::ALL[slot]);
            let j = foot_jacobian_body(
                kin_leg,
                leg_out.q_hip,
                leg_out.q_thigh,
                leg_out.q_calf,
            );
            let tau = -(j.transpose() * f_body);
            out[slot] = Some([tau.x, tau.y, tau.z]);
        }
        out
    }

    pub fn set_full_centroidal_mpc_config(&mut self, cfg: FullCentroidalMpcConfig) {
        self.full_centroidal_mpc.set_config(cfg);
    }
    pub fn full_centroidal_mpc_config(&self) -> &FullCentroidalMpcConfig {
        self.full_centroidal_mpc.config()
    }

    pub fn body_state(&self) -> &BodyState {
        &self.body_state
    }

    pub fn velocity_cmd(&self) -> VelocityCmd {
        self.cmd
    }
    /// Set the body velocity command (vx / vy / wz in body frame).
    /// Implicitly **clears any active goal-pose mode** so callers that
    /// switch back to velocity control don't see lingering position
    /// tracking. Use [`Self::set_goal_pose_world`] for the absolute
    /// position-tracking path.
    pub fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        self.cmd = cmd;
        self.goal_pose = None;
    }

    pub fn config(&self) -> &GaitConfig {
        &self.cfg
    }

    /// Update `cycle_period_s` in place, WITHOUT resetting the phase
    /// clock (`self.phase_gen`'s `cycle_phase`/`holding` state) the
    /// way [`Self::set_config`] does (it rebuilds `phase_gen` from
    /// scratch via `PhaseGenerator::new`, snapping `cycle_phase` back
    /// to 0 -- fine for a one-off gait-family switch, but it would
    /// glitch the gait if called every cycle to nudge the period).
    /// [`crate::phase::PhaseGenerator::set_config`] itself already has
    /// the right (phase-preserving) semantics; this just routes to it
    /// without going through the whole-`GaitConfig` replacement path.
    /// For the adaptive-period ("impulse scaling" / PLL) investigation
    /// -- see `articara/ref/wbc_comparison.md` Sec.5bl.
    pub fn set_cycle_period_s(&mut self, period_s: f64) {
        self.cfg.cycle_period_s = period_s.max(0.05);
        self.phase_gen.set_config(self.cfg.clone());
    }

    /// Update `max_step_length_m` in place. Read live from `self.cfg`
    /// by the footstep planner each tick (not part of `phase_gen`'s own
    /// state), so unlike [`Self::set_cycle_period_s`] this needs no
    /// `phase_gen` round-trip -- a plain field write is enough. For a
    /// smooth startup transient (stride growing in step with a
    /// `cmd_vx` ramp instead of snapping to its full value at t=0),
    /// see `articara/ref/wbc_comparison.md` Sec.5c0.
    pub fn set_max_step_length_m(&mut self, m: f64) {
        self.cfg.max_step_length_m = m.max(0.0);
    }

    /// Apply a new gait config. The MPC's per-tick flags that mirror
    /// fields on `GaitConfig` (currently A3 friction-cone-soft +
    /// slack penalty) are pushed into the live MPC config here so a
    /// single `set_config` round-trip is enough to flip them — the
    /// existing UI/Rhai/test paths already use this method as the
    /// single entry point for config edits.
    pub fn set_config(&mut self, cfg: GaitConfig) {
        self.cfg = cfg.clone();
        self.phase_gen = PhaseGenerator::new(cfg);
        let mut mpc_cfg = self.full_centroidal_mpc.config().clone();
        mpc_cfg.friction_cone_soft = self.cfg.friction_cone_soft;
        mpc_cfg.friction_cone_slack_penalty = self.cfg.friction_cone_slack_penalty;
        let warm_start_was_on = mpc_cfg.warm_start;
        mpc_cfg.warm_start = self.cfg.warm_start;
        // A1: only push q_foot_xy_world to the MPC when the
        // controller-side toggle is on. Off ⇒ keep MPC value at 0
        // (no foot-XY cost) regardless of what `q_foot_xy_world`
        // holds on the gait config (lets the GUI keep the slider
        // value visible while disabled).
        mpc_cfg.q_foot_xy_world = if self.cfg.mpc_optimized_footstep {
            self.cfg.q_foot_xy_world
        } else {
            0.0
        };
        mpc_cfg.foot_xy_cost_body_frame = self.cfg.foot_xy_cost_body_frame;
        self.full_centroidal_mpc.set_config(mpc_cfg);
        // When warm-start is freshly turned off, drop any stale cache
        // so a later re-enable doesn't replay an old trajectory the
        // operator didn't ask for.
        if warm_start_was_on && !self.cfg.warm_start {
            self.full_centroidal_mpc.clear_warm_start();
        }
    }

    pub fn kinematics(&self) -> &KinematicsConfig {
        &self.kin
    }
    pub fn set_kinematics(&mut self, kin: KinematicsConfig) {
        // The MPC config carries its own copy of `kin` for FK; keep them
        // in sync when the host re-tunes the kinematics.
        let mut mpc_cfg = self.full_centroidal_mpc.config().clone();
        mpc_cfg.kinematics = kin.clone();
        self.full_centroidal_mpc.set_config(mpc_cfg);
        self.kin = kin;
    }

    pub fn knee_forward(&self) -> [bool; 4] {
        self.knee_forward
    }
    pub fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        self.knee_forward[crate::controller::slot_of(leg)] = forward;
    }
    pub fn set_knee_pattern(&mut self, pattern: crate::config::KneePattern) {
        self.knee_forward = pattern.to_knee_forward();
    }
    pub fn knee_pattern(&self) -> crate::config::KneePattern {
        crate::config::KneePattern::from_knee_forward(self.knee_forward)
    }

    pub fn capture_point_gain(&self) -> f64 {
        self.k_capture
    }
    pub fn set_capture_point_gain(&mut self, k: f64) {
        self.k_capture = k.max(0.0);
    }

    pub fn bound_fore_aft_placement_gain(&self) -> f64 {
        self.bound_fore_aft_placement_gain
    }
    /// Set the Bound-specific fore-aft foot-placement feedback gain
    /// (see the field's doc comment). Clamped to ≥ 0. `0.0` disables it
    /// (default), leaving the cmd-based Raibert `half` untouched.
    pub fn set_bound_fore_aft_placement_gain(&mut self, k: f64) {
        self.bound_fore_aft_placement_gain = k.max(0.0);
    }

    /// Set the Poincaré/deadbeat pitch foot-placement gains
    /// `(k_angle, k_rate)` (see the field's doc comment). Unlike the
    /// speed-regulator gain these are NOT clamped to ≥ 0 -- the pitch
    /// sign convention is antiphase to `euler_angles()`, so the
    /// stabilizing sign is found empirically. `(0.0, 0.0)` (default)
    /// leaves the foothold untouched.
    pub fn set_bound_pitch_placement_gain(&mut self, k_angle: f64, k_rate: f64) {
        self.bound_pitch_placement_gain = k_angle;
        self.bound_pitch_rate_placement_gain = k_rate;
    }

    /// Set the Sec.5f8 DC-blocker time constant (s) for the pitch
    /// foot-placement (see the field's doc). `0.0` disables it. Clamped
    /// to ≥ 0.
    pub fn set_bound_pitch_placement_dc_tau(&mut self, tau: f64) {
        self.bound_pitch_placement_dc_tau = tau.max(0.0);
    }

    /// Set the Sec.5f9 (P2) tabulated forward-Bound reference orbit
    /// (`[phase, z, pitch, vx, vz, w]` rows, ascending phase). Empty or
    /// `None` clears it.
    pub fn set_bound_tabulated_reference(&mut self, table: Option<Vec<[f64; 6]>>) {
        self.bound_tabulated_reference = table.filter(|t| t.len() >= 2);
    }

    /// Phase-interpolate the tabulated reference at `phase` in [0,1),
    /// returning `(z, pitch, vx, vz, w)`. Linear between the two nearest
    /// rows, wrapping across the period boundary.
    fn sample_tabulated_reference(&self, phase: f64) -> Option<(f64, f64, f64, f64, f64)> {
        let t = self.bound_tabulated_reference.as_ref()?;
        let ph = phase.rem_euclid(1.0);
        // find the first row with phase > ph; interpolate with predecessor
        let n = t.len();
        let mut hi = n;
        for (i, row) in t.iter().enumerate() {
            if row[0] > ph {
                hi = i;
                break;
            }
        }
        let (lo_i, hi_i, span_lo, span_hi) = if hi == 0 {
            // before the first row: wrap with the last row (phase-1)
            (n - 1, 0, t[n - 1][0] - 1.0, t[0][0])
        } else if hi == n {
            // after the last row: wrap to the first (phase+1)
            (n - 1, 0, t[n - 1][0], t[0][0] + 1.0)
        } else {
            (hi - 1, hi, t[hi - 1][0], t[hi][0])
        };
        let denom = (span_hi - span_lo).max(1e-9);
        let frac = ((ph - span_lo) / denom).clamp(0.0, 1.0);
        let a = &t[lo_i];
        let b = &t[hi_i];
        let lerp = |j: usize| a[j] + frac * (b[j] - a[j]);
        Some((lerp(1), lerp(2), lerp(3), lerp(4), lerp(5)))
    }

    /// Read the nonlinear pulse branch parameters `(k_pulse, v_db)`.
    /// `k_pulse = 0` means the pulse branch is inactive — the
    /// controller uses pure linear capture-point.
    pub fn capture_point_pulse(&self) -> (f64, f64) {
        (self.k_capture_pulse, self.v_capture_deadband)
    }
    /// Configure the nonlinear pulse branch of the capture-point
    /// feedback. `k_pulse` is the slope applied to `(|v_err| − v_db)`
    /// for `|v_err| > v_db`; below the deadband the pulse contributes
    /// 0 and the controller falls back to its linear `k_capture` gain
    /// alone. Both are clamped to ≥ 0. See
    /// [`crate::mpc_controller::capture_point_step`].
    pub fn set_capture_point_pulse(&mut self, k_pulse: f64, v_db: f64) {
        self.k_capture_pulse = k_pulse.max(0.0);
        self.v_capture_deadband = v_db.max(0.0);
    }

    pub fn set_body_state_observed(
        &mut self,
        v_world: Vector3<f64>,
        omega_world: Vector3<f64>,
    ) {
        self.v_observed_world = v_world;
        self.omega_observed_world = omega_world;
    }

    pub fn set_body_pose_observed(
        &mut self,
        world_yaw: f64,
        world_position: Vector3<f64>,
    ) {
        self.body_state.world_yaw = world_yaw;
        self.body_state.world_position = world_position;
    }

    /// Feed the observed base roll/pitch (world-frame ZYX Euler, rad),
    /// consumed only by the Poincaré/deadbeat pitch foot-placement
    /// (Sec.5f6). Optional -- callers that never set it keep the prior
    /// zero-attitude behaviour, so this is non-breaking.
    pub fn set_body_attitude_observed(&mut self, roll: f64, pitch: f64) {
        self.roll_observed = roll;
        self.pitch_observed = pitch;
    }

    pub fn reset(&mut self) {
        self.body_state = BodyState::new();
        self.phase_gen.reset();
        self.cmd = VelocityCmd::zero();
        self.last_solution = None;
        self.last_solution_compat = None;
        self.mpc_solve_accumulator_s = f64::INFINITY;
        // Discard any in-flight background solve so its (now stale) result
        // can't repopulate the just-cleared solution.
        self.mpc_worker.reset();
        self.swing_locked_touch_down_body = [None; N_FEET];
        self.prev_leg_is_stance = [true; N_FEET];
    }

    pub fn tick(&mut self, dt: f64) -> ControllerOutput {
        // Goal-pose mode (legged_control parity): recompute the
        // velocity command from the absolute goal + observed body
        // pose, so a disturbance that drifts the body off-track gets
        // converted into a non-zero v_y_cmd pointing back at the goal.
        if let Some(goal) = self.goal_pose {
            self.cmd = velocity_cmd_for_goal(
                goal,
                self.body_state.world_position,
                self.body_state.world_yaw,
            );
        }
        self.phase_gen.advance(dt, &self.cmd);
        self.body_state.integrate(&self.cmd, dt);

        let v_obs_body = world_to_body_horizontal(
            self.v_observed_world,
            self.body_state.world_yaw,
        );
        // EMA of measured fore-aft velocity for the Raibert neutral
        // point (Sec.5c7). tau ≈ 0.15 s (~one cycle period) smooths
        // per-tick observer noise while tracking speed changes over a
        // stride. Only consumed when the bound foot-placement gain is
        // on, but updated unconditionally so it's warm the moment it is.
        {
            let tau = 0.15;
            let alpha = (dt / (tau + dt)).clamp(0.0, 1.0);
            self.v_fore_aft_filtered += alpha * (v_obs_body.x - self.v_fore_aft_filtered);
        }

        // Sec.5f8 pitch foot-placement shift, computed ONCE per tick (the
        // shift is leg-independent: it depends only on the global base
        // pitch state). Orbit-relative deviation (subtract the trim
        // nominal at the current phase) plus an optional DC-blocker that
        // removes the residual persistent forward bias (which otherwise
        // drags the body backward, Sec.5f7). Every leg's footstep reads
        // the resulting `self.pitch_placement_shift`.
        if self.bound_pitch_placement_gain != 0.0 || self.bound_pitch_rate_placement_gain != 0.0 {
            // Nominal for the orbit-relative deadbeat. The closed-form trim
            // orbit is used when active; otherwise absolute (0,0). NOTE
            // (Sec.5f9/P2): subtracting the *tabulated* SRBD orbit's
            // pitch_rate was tried and made it WORSE (tumble at ~5s) --
            // the planar SRBD nominal ≠ the real 3D Go2 pitch_rate, so the
            // "deviation" is polluted by model mismatch and the deadbeat
            // loses its stabilizing action. Instead the deadbeat keeps its
            // full (absolute/trim) stabilizing action and the DC-blocker
            // (model-free) removes the backward-drag DC, while the
            // tabulated reference supplies the forward drive.
            let cur_phase = self.phase_gen.cycle_phase();
            let (nom_pitch, nom_pitch_rate) = if let Some(trim) = self.bound_trim_config() {
                let s = trim.sample(cur_phase);
                (s.pitch, s.pitch_rate)
            } else {
                (0.0, 0.0)
            };
            let pitch_err = self.pitch_observed - nom_pitch;
            let pitch_rate_err = self.omega_observed_world.y - nom_pitch_rate;
            let raw = self.bound_pitch_placement_gain * pitch_err
                + self.bound_pitch_rate_placement_gain * pitch_rate_err;
            if self.bound_pitch_placement_dc_tau > 0.0 {
                let alpha = (dt / (self.bound_pitch_placement_dc_tau + dt)).clamp(0.0, 1.0);
                self.pitch_placement_shift_dc += alpha * (raw - self.pitch_placement_shift_dc);
                self.pitch_placement_shift = raw - self.pitch_placement_shift_dc;
            } else {
                self.pitch_placement_shift = raw;
            }
        } else {
            self.pitch_placement_shift = 0.0;
            self.pitch_placement_shift_dc = 0.0;
        }

        let v_cmd = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
        let v_err_body = v_obs_body - v_cmd;

        let phases = self.phase_gen.legs();
        let swing_duration = self.cfg.cycle_period_s * (1.0 - self.cfg.duty_factor);

        // Pre-pass (Sec.5d3): latch each leg's MPC-predicted foothold at
        // its stance→swing transition, THEN — if bound_symmetric_foothold
        // is on — symmetrize each L/R pair before the main loop consumes
        // the latches. Done as a separate pass (not inline in the main
        // loop) because symmetrizing a pair needs both legs' freshly-
        // latched values at once, and a pair's two legs are visited in
        // separate loop iterations. Sec.5d2: the MPC's independent per-
        // leg foot-XY optimization produced asymmetric footholds that
        // rolled the body over during the aerial phase.
        if self.cfg.mpc_optimized_footstep {
            for ps in phases.iter() {
                let slot = crate::controller::slot_of(ps.leg);
                if !ps.is_stance {
                    if self.prev_leg_is_stance[slot] {
                        self.swing_locked_touch_down_body[slot] =
                            self.mpc_predicted_swing_target_body(ps.leg, 0.0, swing_duration);
                    }
                } else {
                    self.swing_locked_touch_down_body[slot] = None;
                }
            }
            if self.cfg.bound_symmetric_foothold {
                // Pairs: front (FL=0, FR=1), rear (RL=2, RR=3). Symmetrize
                // about the sagittal plane: x,z averaged; y mirrored
                // (y_L = +a, y_R = −a with a = (y_L − y_R)/2).
                for &(l, r) in &[(0usize, 1usize), (2usize, 3usize)] {
                    if let (Some(mut tl), Some(mut tr)) =
                        (self.swing_locked_touch_down_body[l], self.swing_locked_touch_down_body[r])
                    {
                        let x = 0.5 * (tl.x + tr.x);
                        let z = 0.5 * (tl.z + tr.z);
                        let a = 0.5 * (tl.y - tr.y);
                        tl.x = x; tl.z = z; tl.y = a;
                        tr.x = x; tr.z = z; tr.y = -a;
                        self.swing_locked_touch_down_body[l] = Some(tl);
                        self.swing_locked_touch_down_body[r] = Some(tr);
                    }
                }
            }
        }

        let mut legs: [Option<LegOutput>; 4] = [None, None, None, None];
        for ps in phases.iter() {
            let kin_leg = self.kin.leg(ps.leg);
            let mut footstep = self.compute_mpc_footstep(kin_leg, &v_err_body);
            // **A1 closed-loop foothold (lock-at-swing-entry)**.
            // When the MPC-optimised footstep mode is on, capture the
            // MPC's predicted foot position **once** at the stance →
            // swing transition and reuse it for the whole swing. The
            // v3 bench showed that recomputing per-tick (the MPC
            // solves every 30 ms while sub_fraction advances every 2
            // ms) produces a moving target that the swing curve
            // chases until the body tips. Locking at swing entry
            // removes the wobble entirely: the foot aims at a fixed
            // point inside the body frame chosen with the freshest
            // possible MPC prediction, and the stance no-slip pin
            // (which uses cmd-extrap, separate axis) keeps the body
            // honest during stance.
            let slot = crate::controller::slot_of(ps.leg);
            // The latch (+ optional pair symmetrization) is done in the
            // pre-pass above; here we just consume the locked value for
            // swing legs.
            if self.cfg.mpc_optimized_footstep && !ps.is_stance {
                if let Some(td_body) = self.swing_locked_touch_down_body[slot] {
                    footstep.touch_down = td_body;
                }
            }
            self.prev_leg_is_stance[slot] = ps.is_stance;
            let target = if ps.is_stance {
                footstep.stance_at(ps.sub_fraction)
            } else {
                let swing_h = effective_swing_height(self.cfg.swing_height_m, &self.cmd);
                swing_position(
                    footstep.lift_off,
                    footstep.touch_down,
                    swing_h,
                    ps.sub_fraction,
                )
            };
            let knee_fwd = self.knee_forward[crate::controller::slot_of(ps.leg)];
            let sol = solve_leg_ik(kin_leg, target, knee_fwd);
            let reachable = matches!(sol, LegIkSolution::Reached { .. });
            let (h, t, c) = sol.angles();
            legs[crate::controller::slot_of(ps.leg)] = Some(make_leg_output(
                ps.leg, kin_leg, *ps, footstep, target, h, t, c, reachable,
            ));
        }
        let output = ControllerOutput {
            legs: legs.map(|x| x.expect("all four legs filled by phase loop")),
            body_state: self.body_state,
        };

        // Adopt any solution the background worker finished (async only).
        if let Some(sol) = self.mpc_worker.poll() {
            self.last_solution_compat = Some(to_compat_mpc_solution_full(&sol));
            self.last_solution = Some(sol);
        }
        let dt_per_step = self.full_centroidal_mpc.config().dt_per_step;
        self.mpc_solve_accumulator_s += dt;
        if self.mpc_solve_accumulator_s >= dt_per_step {
            if self.async_mpc {
                // Off-thread solve: queue only when idle, keep the previous
                // solution (ZOH) until the fresh one lands. The cloned
                // solver carries the current SQP warm-start cache. This is
                // what keeps the GUI responsive under the ≈0.4 s solve.
                if !self.mpc_worker.is_busy() {
                    let (s_now, reference, contact) =
                        self.build_full_centroidal_inputs(&output);
                    let mut solver = self.full_centroidal_mpc.clone();
                    self.mpc_worker.submit(move || {
                        solver.solve(s_now, &reference, &contact)
                    });
                    self.mpc_solve_accumulator_s = 0.0;
                }
            } else {
                // Synchronous solve (default): deterministic, used by
                // headless benchmarks / tests / hardware. Blocks the caller.
                let (s_now, reference, contact) =
                    self.build_full_centroidal_inputs(&output);
                let sol = self.full_centroidal_mpc.solve(s_now, &reference, &contact);
                self.last_solution_compat = Some(to_compat_mpc_solution_full(&sol));
                self.last_solution = Some(sol);
                self.mpc_solve_accumulator_s = 0.0;
            }
        }

        output
    }

    fn build_full_centroidal_inputs(
        &self,
        output: &ControllerOutput,
    ) -> (
        FullCentroidalState,
        FullCentroidalReference,
        FullCentroidalContactSchedule,
    ) {
        let cfg = self.full_centroidal_mpc.config().clone();
        let n = cfg.horizon_steps;

        // Current joint_q from the IK output (12 entries, FL/FR/RL/RR ×
        // [hip, thigh, calf]). These feed the per-node FK in the MPC so
        // the moment arm at step 0 matches what the legs are actually
        // doing.
        let mut joint_q_now = [0.0_f64; N_LEG_JOINTS];
        for slot in 0..N_FEET {
            let leg = &output.legs[slot];
            joint_q_now[3 * slot] = leg.q_hip;
            joint_q_now[3 * slot + 1] = leg.q_thigh;
            joint_q_now[3 * slot + 2] = leg.q_calf;
        }

        let s_now = FullCentroidalState {
            v_com_world: self.v_observed_world,
            angular_velocity_world: self.omega_observed_world,
            base_pos_world: self.body_state.world_position,
            base_euler_zyx: Vector3::new(0.0, 0.0, self.body_state.world_yaw),
            joint_q: joint_q_now,
        };

        // Build reference: cmd-velocity integrated body trajectory +
        // held joint_q + joint_v=0 + gravity-balanced GRF.
        //
        // joint_q held constant over the horizon (D3.3.5a simplification
        // — design choice (a)). The MPC's stance no-slip constraint will
        // still produce non-zero joint_v as needed to keep stance feet
        // pinned; the cost just doesn't bias swing legs to follow the
        // open-loop footstep trajectory in this revision.
        let v_world_cmd = body_to_world_horizontal(
            Vector3::new(self.cmd.vx, self.cmd.vy, 0.0),
            self.body_state.world_yaw,
        );
        let dt_per_step = cfg.dt_per_step;
        let stance_now: [bool; N_FEET] = [
            output.legs[0].phase.is_stance,
            output.legs[1].phase.is_stance,
            output.legs[2].phase.is_stance,
            output.legs[3].phase.is_stance,
        ];
        let holding = self.cmd.is_zero();

        // Per-step contact schedule. Two paths:
        //
        // - Legacy (D3.3.5a): step 0 = observed stance, k≥1 = `duty > 0.5
        //   ? all stance : all swing`. Cheap proxy with no per-leg phase
        //   awareness; carried because the existing benchmark rows
        //   (`FullC default / h20 sqp3 / h10 sqp5`) were tuned against
        //   it and the contact schedule mismatch is part of their
        //   character.
        // - legged_control parity (D3.3.5b): step k's per-leg stance is
        //   derived from the projected per-leg phase
        //   `(cycle_phase_now + k·dt_per_step / cycle_period + offset) mod 1`,
        //   matching the OCS2 `SwitchedModelReferenceManager`. Each
        //   swing-leg-step also carries a planned vertical foot velocity
        //   (from [`swing_vz_world`]) so the MPC's NormalVelocity-equivalent
        //   equality constraint has something to track.
        let mut contact = FullCentroidalContactSchedule {
            is_stance: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            swing_z_velocity: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            stance_f_max: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            foot_xy_target_body_offset: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        };
        // Per-(leg, step) stance sub-fraction, kept alongside the
        // schedule so the C1 GRF-reference ramp can look up "how far
        // through stance is this leg at step k". Filled only when the
        // leg is in stance (swing entries are unused).
        let mut stance_sub_fractions: [Vec<f64>; N_FEET] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        // Per-(leg, step) phase sub-fraction regardless of stance/swing
        // (unlike `stance_sub_fractions`, which zeroes swing entries) —
        // feeds `dynamic_joint_q_reference`'s per-step swing/stance foot
        // sampling below.
        let mut phase_sub_fractions: [Vec<f64>; N_FEET] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let cycle_phase_now = self.phase_gen.cycle_phase();
        let cycle_period = self.cfg.cycle_period_s.max(1e-6);
        // Front-pair duty. The rear pair may run a different one
        // (`duty_factor_rear_scale`), so anything inside the per-leg loop
        // must use `self.cfg.duty_for_slot(leg)`; this alias is only for the
        // quantities that are genuinely global or FL-referenced.
        let duty = self.cfg.duty_factor.clamp(1e-6, 1.0 - 1e-6);
        let swing_duration = cycle_period * (1.0 - duty);
        let swing_h = effective_swing_height(self.cfg.swing_height_m, &self.cmd);
        let leg_phase_offsets: [f64; N_FEET] = {
            let mut arr = [0.0_f64; N_FEET];
            for (leg, off) in self.cfg.gait_type.phase_offsets() {
                arr[crate::controller::slot_of(leg)] = off;
            }
            arr
        };
        for leg in 0..N_FEET {
            let duty = self.cfg.duty_for_slot(leg);
            for k in 0..n {
                let (in_stance, sub_frac, in_swing) = if holding {
                    // Holding (zero cmd): every leg is in mid-stance,
                    // so the C1 GRF ramp picks weight = 1.0 and the
                    // legacy even-split math is preserved exactly.
                    (true, 0.5_f64, false)
                } else if self.legged_control_parity {
                    // Project the cycle phase forward by k·dt_per_step.
                    // The k=0 row keeps the observed stance flag — the
                    // system is in that state right now and the no-slip
                    // equality at step 0 must not conflict with reality.
                    // For swing v_z the observed sub_fraction is used so
                    // the planned velocity is continuous with the
                    // foot's current motion.
                    if k == 0 {
                        let phase = output.legs[leg].phase;
                        (phase.is_stance, phase.sub_fraction, !phase.is_stance)
                    } else {
                        let t = k as f64 * dt_per_step;
                        let cycle_phase_k =
                            (cycle_phase_now + t / cycle_period).rem_euclid(1.0);
                        let pos = (cycle_phase_k + leg_phase_offsets[leg]).rem_euclid(1.0);
                        if pos < duty {
                            (true, pos / duty, false)
                        } else {
                            (false, (pos - duty) / (1.0 - duty), true)
                        }
                    }
                } else if k == 0 {
                    // Legacy path has no per-step phase info; pretend
                    // mid-stance so C1 weight is 1.0 (transition_fraction
                    // is parity-only by construction).
                    (stance_now[leg], 0.5, false)
                } else {
                    (duty > 0.5, 0.5, false)
                };
                contact.is_stance[leg].push(in_stance);
                let v_z = if in_swing && self.legged_control_parity {
                    swing_vz_world(swing_h, sub_frac, swing_duration, 0.0, 0.0)
                } else {
                    0.0
                };
                contact.swing_z_velocity[leg].push(v_z);
                // C1: stash the stance sub-fraction so the GRF
                // reference loop below can apply the transition ramp.
                // Swing entries get 0.0 (unused).
                stance_sub_fractions[leg].push(if in_stance { sub_frac } else { 0.0 });
                phase_sub_fractions[leg].push(sub_frac);
                // C1-2: per-(leg, k) f_z upper bound. When the
                // constraint-side ramp is enabled (and we're on the
                // parity path with a non-zero transition_fraction),
                // tighten the bound to `weight · cfg.max_normal_force`.
                // Otherwise INFINITY ⇒ the global f_max applies
                // unchanged (backward-compat).
                let f_max_cell = if in_stance
                    && self.legged_control_parity
                    && self.cfg.transition_enforce_constraint
                    && self.cfg.transition_fraction > 0.0
                {
                    let mpc_f_max = cfg.max_normal_force.max(0.0);
                    let w = crate::config::stance_weight_at(
                        sub_frac,
                        self.cfg.transition_fraction,
                    );
                    mpc_f_max * w
                } else {
                    f64::INFINITY
                };
                contact.stance_f_max[leg].push(f_max_cell);
                // A1: foot XY target — populated below per touchdown
                // step when `mpc_optimized_footstep` is on. Push None
                // here as the per-step default so the length asserts
                // match `n`.
                contact.foot_xy_target_body_offset[leg].push(None);
            }
        }

        // A1: when MPC-optimised footstep is enabled, fill the
        // touchdown-step XY target for each leg from the existing
        // footstep planner. The MPC's foot-XY soft cost then drives
        // the swing-leg joint trajectory to land at that target,
        // self-consistently considering the predicted base motion.
        //
        // Touchdown step = first step where this leg is in stance
        // *and* the previous step was in swing (or it's step 0 and
        // the leg is starting stance). Open-loop the planner already
        // computes a body-frame `touch_down` per leg from the Raibert
        // + cap-pt formula; we rotate that into the world frame at
        // the predicted yaw (= ref_traj yaw at that step) and pass it
        // as the cost target. Skipping when the cmd is zero (holding
        // pose) since there's no swing then.
        if self.cfg.mpc_optimized_footstep && !holding {
            // Body-frame velocity error from the latest observed CoM
            // velocity and the current cmd.
            let v_obs_body_now = world_to_body_horizontal(
                self.v_observed_world,
                self.body_state.world_yaw,
            );
            let v_cmd_now = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
            let v_err_body = v_obs_body_now - v_cmd_now;
            for leg in 0..N_FEET {
                let kin_leg = self.kin.leg(LegId::ALL[leg]);
                // Find the first stance step preceded by swing.
                let mut touchdown_k: Option<usize> = None;
                let mut prev_stance: Option<bool> = None;
                for k in 0..n {
                    let s_k = contact.is_stance[leg][k];
                    let is_touchdown = match prev_stance {
                        Some(prev) => s_k && !prev,
                        None => false,
                    };
                    if is_touchdown {
                        touchdown_k = Some(k);
                        break;
                    }
                    prev_stance = Some(s_k);
                }
                let Some(k_td) = touchdown_k else { continue; };

                // Pass the **body-frame** foot offset (Raibert + cap-pt)
                // to the MPC. The MPC will combine it with its iterated
                // `ref_traj.states[k_td].base_pos_world` each SQP step
                // to form the world target, so when the predicted base
                // drifts from the open-loop cmd extrapolation under
                // disturbance the foothold target follows along
                // automatically.
                let footstep = self.compute_mpc_footstep(kin_leg, &v_err_body);
                let touch_down_body = footstep.touch_down;
                contact.foot_xy_target_body_offset[leg][k_td] =
                    Some([touch_down_body.x, touch_down_body.y]);
                let _ = dt_per_step; // kept for symmetry with neighbouring blocks
            }
        }

        // β: when parity + nominal-q_ref is on, build the URDF nominal
        // stance pose once (3R analytical IK of each leg's
        // `kin.nominal_foot_body`) and use that as the joint_q
        // tracking reference for every horizon step. This matches
        // legged_control's `DEFAULT_JOINT_STATE` design — the swing
        // leg's cost biases it back toward the standing pose rather
        // than tracking whatever the leg happens to be doing.
        let nominal_joint_q: Option<[f64; N_LEG_JOINTS]> =
            if self.legged_control_parity && self.parity_use_nominal_q_ref {
                let mut q = [0.0_f64; N_LEG_JOINTS];
                for slot in 0..N_FEET {
                    let kin = self.kin.leg(LegId::ALL[slot]);
                    let knee_fwd = self.knee_forward[slot];
                    let sol = solve_leg_ik(kin, kin.nominal_foot_body, knee_fwd);
                    let (h, th, c) = sol.angles();
                    q[3 * slot] = h;
                    q[3 * slot + 1] = th;
                    q[3 * slot + 2] = c;
                }
                Some(q)
            } else {
                None
            };

        // γ: when `dynamic_joint_q_reference` is on (and parity, since
        // that's what makes `phase_sub_fractions`/`contact.is_stance`
        // meaningful per-leg past step 0), sample each leg's open-loop
        // foot curve — the same `Footstep::stance_at` / `swing_position`
        // + `solve_leg_ik` pattern `tick()` uses for the *current* tick
        // — at every horizon step's *projected* phase instead. This is
        // the D3.3.5a reversal: the joint_q reference becomes a real
        // per-step trajectory instead of a flat hold, so the MPC's
        // (already-generic, already-existing) per-node joint_q cost
        // actually has something meaningful to track for the swing leg.
        //
        // The `Footstep` itself (Raibert + cap-pt touchdown) is computed
        // once per leg, open-loop, exactly like the A1 block above —
        // not from the MPC's own in-flight solve, so there's no
        // intra-solve circularity.
        let dynamic_footsteps: Option<[Footstep; N_FEET]> =
            if self.dynamic_joint_q_reference && self.legged_control_parity && !holding {
                let v_obs_body_now = world_to_body_horizontal(
                    self.v_observed_world,
                    self.body_state.world_yaw,
                );
                let v_cmd_now = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
                let v_err_body_now = v_obs_body_now - v_cmd_now;
                let mut steps =
                    [Footstep { lift_off: Vector3::zeros(), touch_down: Vector3::zeros() }; N_FEET];
                for slot in 0..N_FEET {
                    let kin = self.kin.leg(LegId::ALL[slot]);
                    steps[slot] = self.compute_mpc_footstep(kin, &v_err_body_now);
                }
                Some(steps)
            } else {
                None
            };

        // Bound trim reference (Sec.5bb/5bc): a closed-form periodic
        // pitch/fore-aft-GRF profile for Bound's front-pair/rear-pair
        // stance, replacing the flat zero-pitch/zero-Fx hold every
        // gait (including Bound, until now) has used. `None` for
        // every other gait, and for Bound while holding (cmd==0) --
        // an oscillating reference makes no sense standing still.
        let fl_slot = crate::controller::slot_of(LegId::FL);
        // Sec.5bb/5bc closed-form pitch/Fx orbit; `None` for other gaits
        // and while holding. Built via the shared helper so the Sec.5f8
        // orbit-relative foot-placement samples the identical nominal.
        let bound_trim: Option<BoundTrimConfig> = self.bound_trim_config();

        // Per-step reference state + input. Body pose integrates the cmd
        // velocity; joint_q held (or set to nominal pose when β is on);
        // gravity distributed across stance legs for the GRF reference
        // (the QP deviates as needed for the cost and constraints).
        let mut ref_states = Vec::with_capacity(n);
        let mut ref_inputs = Vec::with_capacity(n);
        for k in 0..n {
            let t = (k + 1) as f64 * dt_per_step;
            let mut sk = s_now;
            sk.v_com_world = v_world_cmd;
            sk.angular_velocity_world = Vector3::new(0.0, 0.0, self.cmd.wz);
            sk.base_pos_world = s_now.base_pos_world + v_world_cmd * t;
            sk.base_euler_zyx.z = s_now.base_euler_zyx.z + self.cmd.wz * t;
            // Reconstruct this step's global cycle phase from FL's own
            // projected phase (FL's `phase_offsets()` entry is 0.0 for
            // Bound, so its own `pos` *is* the global cycle phase --
            // no separate phase math needed, stays consistent with
            // whatever `contact`/`phase_sub_fractions` already decided
            // for this step).
            let fl_stance = contact.is_stance[fl_slot][k];
            let fl_sub = phase_sub_fractions[fl_slot][k];
            let cycle_phase_k = if fl_stance { fl_sub * duty } else { duty + fl_sub * (1.0 - duty) };
            let trim_sample = bound_trim.map(|trim| trim.sample(cycle_phase_k));
            // Sec.5f9 (P2): a tabulated feasible forward-Bound orbit takes
            // priority over the closed-form trim. It injects the FULL
            // consistent reference (height z, pitch, forward vx, vertical
            // vz, pitch-rate w) at this step's phase -- the missing piece
            // §5f7/5f8 found: forward-vx + trim-pitch + flat-height were
            // mutually infeasible, so the MPC gave vx≈0. A consistent
            // orbit gives it a trackable forward target. Straight bound
            // (yaw≈0) => world x = forward. (GRF reference below still uses
            // the trim's f_x when a trim is set; the table has no forces,
            // so the MPC optimizes GRF around the gravity split otherwise.)
            if let Some((z, pitch, vx, vz, w)) = self.sample_tabulated_reference(cycle_phase_k) {
                sk.base_euler_zyx.y = pitch;
                sk.v_com_world.x = vx;
                sk.v_com_world.z = vz;
                sk.base_pos_world.z = z;
                sk.angular_velocity_world.y = w;
            } else if let Some(sample) = bound_trim.map(|trim| trim.sample(cycle_phase_k)) {
                sk.base_euler_zyx.y = sample.pitch;
                // Sec.5d4: feed the ballistic vertical-bounce velocity
                // into the reference (opt-in). NOTE: verified against
                // MIT Cheetah 3 / Mini Cheetah, this is AWAY from their
                // design -- they command z-velocity=0 and let the bounce
                // emerge. Kept flag-gated (default off) so the on/off
                // A/B is measurable; off = MIT-aligned flat reference.
                if self.cfg.bound_trim_vertical_reference {
                    sk.v_com_world.z = sample.com_z_velocity;
                }
            }
            if let Some(footsteps) = &dynamic_footsteps {
                // γ: per-(leg, k) dynamic joint_q — takes priority over
                // the β nominal-pose override (both are opt-in and
                // mutually exclusive in practice; γ is the more complete
                // behaviour when both happen to be enabled).
                for slot in 0..N_FEET {
                    let kin = self.kin.leg(LegId::ALL[slot]);
                    let sub_frac = phase_sub_fractions[slot][k];
                    let target = if contact.is_stance[slot][k] {
                        footsteps[slot].stance_at(sub_frac)
                    } else {
                        swing_position(footsteps[slot].lift_off, footsteps[slot].touch_down, swing_h, sub_frac)
                    };
                    let knee_fwd = self.knee_forward[slot];
                    let sol = solve_leg_ik(kin, target, knee_fwd);
                    let (h, th, c) = sol.angles();
                    sk.joint_q[3 * slot] = h;
                    sk.joint_q[3 * slot + 1] = th;
                    sk.joint_q[3 * slot + 2] = c;
                }
            } else if let Some(q_nom) = nominal_joint_q {
                sk.joint_q = q_nom;
            }
            ref_states.push(sk);

            // Gravity-balanced GRF reference: total = m·g, split
            // across stance legs at this step.
            //
            // **C1 (transition_fraction > 0)**: each stance leg's share
            // is weighted by `stance_weight_at(sub_frac, tw)` — newly
            // touched-down legs and about-to-lift legs get a smaller
            // share so the MPC's GRF *target* trajectory ramps in /
            // out rather than stepping. This is a soft (cost-side)
            // smoother; the stance no-slip equality still pins the
            // foot regardless of weight. Backward-compat: when
            // `transition_fraction == 0` the weight is always 1.0 so
            // the math reduces to the legacy even split.
            let tw = self.cfg.transition_fraction;
            let mut leg_weights = [0.0_f64; N_FEET];
            let mut total_weight = 0.0_f64;
            for leg in 0..N_FEET {
                if contact.is_stance[leg][k] {
                    let w = crate::config::stance_weight_at(stance_sub_fractions[leg][k], tw);
                    leg_weights[leg] = w;
                    total_weight += w;
                }
            }
            let f_per_unit = if total_weight > 1e-9 {
                cfg.mass_kg * 9.81 / total_weight
            } else {
                0.0
            };
            let mut grfs = [Vector3::zeros(); N_FEET];
            for leg in 0..N_FEET {
                if contact.is_stance[leg][k] {
                    grfs[leg].z = leg_weights[leg] * f_per_unit;
                }
            }
            // Bound trim: fore-aft GRF (Sec.5bb/5bc). Reuses the same
            // `leg_weights`/`total_weight` transition ramp already
            // applied to `grfs[leg].z` above, so the added `F_x` term
            // ramps in/out at pair-switch instants exactly like the
            // gravity split does, rather than stepping. `grfs[leg].x`
            // was never set anywhere before this (always the
            // `Vector3::zeros()` default) -- the MPC has never had any
            // fore-aft GRF signal to chase, for any gait.
            if let Some(sample) = trim_sample {
                if total_weight > 1e-9 {
                    let f_x_total = sample.f_x_per_leg * 2.0;
                    // Vertical GRF reference from the trim's own F_z
                    // (Sec.5d4): for duty<0.5 the trim's `f_z_total =
                    // m·g/(2·duty)` exceeds pure gravity support (m·g),
                    // and that surplus is exactly the impulse the stance
                    // must supply to reverse the flight-phase fall and
                    // bounce the CoM back up. The gravity-only z set
                    // above (`m·g` split) leaves the reference physically
                    // infeasible through the aerial phase (commands "hold
                    // flat height" with only gravity force while 30%
                    // airborne). At duty=0.5, `f_z_total == m·g`, so this
                    // is byte-identical to the gravity split -- the
                    // stock no-flight Bound is unchanged. `f_z_per_leg`
                    // is already computed and, until now, only used for
                    // the WBC's per-foot z; feeding it to the MPC's own
                    // reference is the missing half.
                    let f_z_total = sample.f_z_per_leg * 2.0;
                    for leg in 0..N_FEET {
                        if contact.is_stance[leg][k] {
                            grfs[leg].x = leg_weights[leg] * f_x_total / total_weight;
                            // Vertical GRF surplus for the bounce (opt-in,
                            // Sec.5d4). Off (default) keeps the pure
                            // gravity-split z set above -- which, per the
                            // MIT verification, is the aligned choice
                            // (they don't command a bounce force either).
                            if self.cfg.bound_trim_vertical_reference {
                                grfs[leg].z = leg_weights[leg] * f_z_total / total_weight;
                            }
                        }
                    }
                }
            }
            // Sec.5d7 forward thrust bias: add a constant net forward
            // GRF (world-x ≈ body-forward for straight running) on
            // stance feet, distributed by the same transition-ramp
            // weights. Raises the trimless MIT line's speed ceiling by
            // supplying the forward force the velocity-tracking cost
            // can't at high command. `0.0` (default) is a no-op.
            if self.cfg.bound_fx_thrust_bias != 0.0 && total_weight > 1e-9 {
                // `bound_fx_thrust_rear_frac` splits the push front/rear
                // (gathered-gallop asymmetry). At the 0.5 default both gains
                // are exactly 1.0, so this is bit-identical to the symmetric
                // version that preceded it.
                let frac = self.cfg.bound_fx_thrust_rear_frac.clamp(0.0, 1.0);
                let pair_gain = [
                    2.0 * (1.0 - frac), 2.0 * (1.0 - frac), // FL, FR
                    2.0 * frac, 2.0 * frac,                 // RL, RR
                ];
                for leg in 0..N_FEET {
                    if contact.is_stance[leg][k] {
                        grfs[leg].x += leg_weights[leg] * pair_gain[leg]
                            * self.cfg.bound_fx_thrust_bias / total_weight;
                    }
                }
            }
            ref_inputs.push(FullCentroidalInput {
                grfs_world: grfs,
                joint_v: [0.0; N_LEG_JOINTS],
            });
        }
        let reference = FullCentroidalReference {
            states: ref_states,
            inputs: ref_inputs,
        };

        (s_now, reference, contact)
    }

    /// Footstep planner — identical to the 12-state version. Duplicated
    /// (not delegated) so the two controllers can be evaluated head-to-
    /// head without state leak.
    /// Body-frame correction term derived from the MPC's previously-
    /// predicted base trajectory, used by `compute_mpc_footstep` when
    /// `use_mpc_predicted_footstep` is on.
    ///
    /// Idea: at touchdown time `t = swing_duration`, the body will be
    /// at `predicted_base_world(t)`. The cmd-only Raibert formula
    /// already accounts for `cmd · t` of that motion via the
    /// `v_hip · 0.5·stance_duration` term, so we add only the
    /// **disturbance-driven residual** = `(predicted − current) −
    /// cmd · t`. Rotate that residual into the body frame so it
    /// stacks linearly with the rest of `half`.
    ///
    /// Returns zero when `last_solution` is missing (first tick / MPC
    /// hasn't run yet) so the caller silently falls back to the
    /// cap-pt path.
    /// **A1**: body-frame foot position the MPC predicts for `leg` at
    /// its next touchdown step, derived from
    /// `last_solution.predicted_states[k_td].leg_joint_q` via the
    /// per-leg FK. Returns `None` when A1 is off, no solved solution
    /// is cached yet, or the touchdown step falls outside the
    /// horizon. The result is in the **predicted body frame** at
    /// `k_td`; using it as the swing curve's `touch_down` slot is the
    /// closed-loop completion of A1 — the IK swing target picks up
    /// the MPC's joint plan instead of the open-loop Raibert target.
    fn mpc_predicted_swing_target_body(
        &self,
        leg: LegId,
        sub_fraction: f64,
        swing_duration: f64,
    ) -> Option<Vector3<f64>> {
        let sol = self.last_solution.as_ref()?;
        if !sol.solved {
            return None;
        }
        let dt_per_step = self.full_centroidal_mpc.config().dt_per_step.max(1e-6);
        // Time remaining until this leg touches down (sub_fraction
        // walks 0 → 1 over one swing), rounded to the nearest MPC
        // step and clamped to the available prediction horizon.
        let time_to_td = (1.0 - sub_fraction.clamp(0.0, 1.0)) * swing_duration;
        let k_td_raw = (time_to_td / dt_per_step).round() as usize;
        let k_td = k_td_raw.min(sol.predicted_states.len().saturating_sub(1));
        let slot = crate::controller::slot_of(leg);
        let [hip, thigh, calf] = sol.predicted_states[k_td].leg_joint_q(slot);
        let kin = self.kin.leg(leg);
        let foot_body = forward_leg_kinematics(kin, hip, thigh, calf);
        // Sanity: if the predicted foot has drifted into a
        // non-physical position (z above the base, gross workspace
        // violation), fall through to the open-loop footstep — the
        // MPC's predicted joint_q can be noisy under cold-start.
        if foot_body.z > -0.05 || !foot_body.x.is_finite() || !foot_body.y.is_finite() {
            return None;
        }
        Some(foot_body)
    }

    fn mpc_predicted_footstep_correction(&self) -> Vector3<f64> {
        let sol = match self.last_solution.as_ref() {
            Some(s) if s.solved => s,
            _ => return Vector3::zeros(),
        };
        let mpc_cfg = self.full_centroidal_mpc.config();
        let swing_duration = self.cfg.cycle_period_s * (1.0 - self.cfg.duty_factor);
        let dt_per_step = mpc_cfg.dt_per_step.max(1e-6);
        // Pick the horizon index closest to one swing duration ahead.
        // Clamped to the available prediction window so a horizon
        // shorter than one swing doesn't panic.
        let k_swing = ((swing_duration / dt_per_step).round() as usize).max(1);
        let k = k_swing.min(sol.predicted_states.len().saturating_sub(1));
        let predicted_world = sol.predicted_states[k].base_pos_world;
        let current_world = self.body_state.world_position;
        let delta_world = predicted_world - current_world;

        // Subtract the open-loop cmd motion the Raibert baseline
        // already covers (`v_hip · 0.5·stance_duration` is the
        // cmd-derived `half`). The cmd is in body frame; rotate to
        // world frame at the current yaw.
        let t_lookahead = (k + 1) as f64 * dt_per_step;
        let cmd_world = body_to_world_horizontal(
            Vector3::new(self.cmd.vx, self.cmd.vy, 0.0),
            self.body_state.world_yaw,
        );
        let expected_world = cmd_world * t_lookahead;
        let residual_world = Vector3::new(
            delta_world.x - expected_world.x,
            delta_world.y - expected_world.y,
            0.0,
        );

        // World → body frame so the term composes with Raibert's
        // body-frame `half`.
        world_to_body_horizontal(residual_world, self.body_state.world_yaw)
    }

    /// Build the closed-form Bound trim reference for the current config,
    /// or `None` for other gaits / while holding (Sec.5bb/5bc). Shared by
    /// the per-step MPC reference loop and the Sec.5f8 orbit-relative
    /// pitch foot-placement, so both sample the SAME nominal orbit.
    fn bound_trim_config(&self) -> Option<BoundTrimConfig> {
        if !(self.enable_bound_trim_reference
            && self.cfg.gait_type == GaitType::Bound
            && !self.cmd.is_zero())
        {
            return None;
        }
        let cfg = self.full_centroidal_mpc.config();
        let fl_kin = self.kin.leg(LegId::FL);
        let rl_kin = self.kin.leg(LegId::RL);
        let r_x_front = fl_kin.nominal_foot_body.x;
        let r_x_rear = -rl_kin.nominal_foot_body.x;
        let h0 = -fl_kin.nominal_foot_body.z;
        Some(BoundTrimConfig {
            mass_kg: cfg.mass_kg,
            inertia_yy: cfg.centroidal_inertia_body[(1, 1)],
            r_x: 0.5 * (r_x_front + r_x_rear),
            h0,
            cycle_period_s: self.cfg.cycle_period_s,
            duty_factor: self.cfg.duty_factor,
            friction_mu: cfg.friction_mu,
            sign: -1.0,
            thrust_scale: self.bound_trim_thrust_scale,
            cmd_vx_mps: self.cmd.vx,
            velocity_ripple_fraction: self.bound_trim_velocity_ripple_fraction,
        })
    }

    fn compute_mpc_footstep(
        &self,
        kin: &LegKinematics,
        v_err_body: &Vector3<f64>,
    ) -> Footstep {
        // Per-leg: a rear pair on a shorter duty sweeps the ground for less
        // time, so its Raibert half-stride `v * stance/2` must shrink to
        // match or the planned foothold outruns the stance that has to
        // reach it.
        let stance_duration = self.cfg.cycle_period_s * self.cfg.duty_for(kin.leg);
        let v_body = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
        let omega = Vector3::new(0.0, 0.0, self.cmd.wz);
        let v_hip = v_body + omega.cross(&kin.hip_offset);
        let mut half = v_hip * (0.5 * stance_duration);

        let feedback_enabled = !self.cmd.is_zero();

        // legged_control-style path: drop cap-pt and instead derive
        // the foothold correction from the MPC's predicted base
        // displacement over one swing duration. The MPC has already
        // planned how the GRF + state cost will pull the body back
        // from a disturbance, so the predicted base at touchdown is
        // "where the body will be after the MPC's own recovery
        // response" — much more informative than `k · v_err` linear
        // extrapolation from now. Mirrors OCS2 SwingTrajectoryPlanner.
        let closed_loop = if self.use_mpc_predicted_footstep && feedback_enabled {
            self.mpc_predicted_footstep_correction()
        } else {
            // Closed-loop foothold shift in the disturbance direction.
            // Uses the linear `k_capture · v_err` + deadband-gated pulse
            // branch from [`crate::mpc_controller::capture_point_step`]
            // (η-2 experiment): the pulse lets the swing leg commit a
            // larger lateral foothold for real pushes while keeping the
            // small-v_err response gentle so cycle-noise on `v_err_y`
            // can't accumulate into a cross-axis drift.
            let mut feedback = Vector3::zeros();
            if feedback_enabled {
                feedback.x = crate::mpc_controller::capture_point_step(
                    v_err_body.x,
                    self.k_capture,
                    self.k_capture_pulse,
                    self.v_capture_deadband,
                );
                feedback.y = crate::mpc_controller::capture_point_step(
                    v_err_body.y,
                    self.k_capture,
                    self.k_capture_pulse,
                    self.v_capture_deadband,
                );
            }
            let horizon_weight = 1.0 / HORIZON_STEPS as f64;
            let mut horizon_bias = Vector3::zeros();
            if feedback_enabled {
                horizon_bias.x = horizon_weight * feedback.x;
                horizon_bias.y = horizon_weight * feedback.y;
            }
            feedback + horizon_bias
        };

        // Bound-specific fore-aft (x-only) Raibert speed regulator.
        // (Sec.5c7) The full Raibert running form is
        //   x_foot = ẋ·T_st/2 + k·(ẋ − ẋ_des),
        // where the NEUTRAL term `ẋ·T_st/2` uses the MEASURED speed ẋ
        // (filtered) -- placing the foot at the no-net-accel point for
        // the speed the robot is actually going, which is stable
        // regardless of whether the command is reachable. Sec.5c6's
        // first attempt kept the cmd-based neutral (`half.x` from
        // `v_hip`) and only added `k·v_err`; when the command exceeded
        // the achievable speed the neutral was chronically over-placed
        // and the feedback destabilized. Here we OVERRIDE the fore-aft
        // neutral with the filtered measured speed and add the
        // command-tracking feedback on top. x-ONLY (Sec.5bt: the
        // generic x+y `k_capture` rolled the body on lateral noise).
        if feedback_enabled && self.bound_fore_aft_placement_gain != 0.0 {
            let v_filt = self.v_fore_aft_filtered;
            half.x = v_filt * (0.5 * stance_duration)
                + self.bound_fore_aft_placement_gain * (v_filt - self.cmd.vx);
        }

        let raw_half = half + closed_loop;
        let mut combined = raw_half;
        let min_x = MIN_HALF_FRACTION * half.x;
        let min_y = MIN_HALF_FRACTION * half.y;
        if half.x > 0.0 && combined.x < min_x {
            combined.x = min_x;
        } else if half.x < 0.0 && combined.x > min_x {
            combined.x = min_x;
        }
        if half.y > 0.0 && combined.y < min_y {
            combined.y = min_y;
        } else if half.y < 0.0 && combined.y > min_y {
            combined.y = min_y;
        }
        half = combined;
        // `max_step_length_rear_scale` lets the rear pair take a longer
        // stride than the front (gathered-gallop asymmetry). Since the pairs
        // tile the cycle, v_max = (stride_front + stride_rear)/T, so scaling
        // only the rear raises the ceiling without touching the FRONT pair's
        // swing speed -- the thing that breaks when max_step_length_m is
        // raised uniformly. At the 1.0 default this is the symmetric value.
        let step_scale = match kin.leg {
            LegId::RL | LegId::RR => self.cfg.max_step_length_rear_scale.max(0.0),
            _ => 1.0,
        };
        let max_half = 0.5 * self.cfg.max_step_length_m * step_scale;
        let mag = half.norm();
        if mag > max_half && mag > 0.0 {
            half *= max_half / mag;
        }

        // Poincaré/deadbeat pitch foot-placement (Sec.5f6/5f7). Shift the
        // TOUCH_DOWN fore-aft by the pitch error so the upcoming stance's
        // GRF moment nulls the accumulated pitch momentum. Applied to
        // touch_down ONLY (not the symmetric `half`) so it does NOT bias
        // the step CENTER: Sec.5f7 found that folding it into `half`
        // shifted lift_off backward too, biasing net fore-aft travel into
        // a persistent backward drift the speed regulator couldn't
        // reverse. Leaving lift_off at the speed-neutral point and moving
        // only the landing target decouples attitude control from the
        // fore-aft drive. Independent of `feedback_enabled` (attitude must
        // be stabilized even at zero velocity command); clamped to the
        // same max-step envelope.
        let mut touch_down = kin.nominal_foot_body + half;
        // Sec.5f6/5f8 Poincaré/deadbeat pitch foot-placement. The shift
        // (orbit-relative deviation, optionally DC-blocked) is computed
        // once per tick in `tick()` -- it is leg-independent -- and only
        // applied to TOUCH_DOWN here (not the symmetric `half`, Sec.5f7)
        // so it does not bias the step center. Clamped to the same
        // max-step envelope.
        if self.pitch_placement_shift != 0.0 {
            touch_down.x += self.pitch_placement_shift;
            let td_rel = touch_down - kin.nominal_foot_body;
            let td_mag = td_rel.norm();
            if td_mag > max_half && td_mag > 0.0 {
                touch_down = kin.nominal_foot_body + td_rel * (max_half / td_mag);
            }
        }
        Footstep {
            lift_off: kin.nominal_foot_body - half,
            touch_down,
        }
    }
}

/// Convert an absolute world-frame goal pose into the instantaneous
/// body-frame velocity command that drives the body toward the goal in
/// approximately straight-line fashion, saturated at the per-axis
/// limits. Matches legged_control's `goalToTargetTrajectories` shape
/// (line 33-51 + 54-68 of `target_trajectories_publisher.cpp`):
///
/// 1. Compute the world-frame error `(dx, dy, dψ)`.
/// 2. Estimate `t_to_target` so that no axis exceeds its max rate —
///    `max(‖(dx,dy)‖/max_v, |dψ|/max_wz, eps)` with a 50 ms floor to
///    avoid division by zero near the goal.
/// 3. Divide the error by `t_to_target` and clamp each axis at its
///    saturation limit. World-frame velocities are then rotated into
///    the body frame using the observed yaw.
/// 4. If within the configured tolerances, emit
///    [`VelocityCmd::zero`] so the phase generator holds in stance.
pub fn velocity_cmd_for_goal(
    goal: GoalPoseWorld,
    current_pos_world: Vector3<f64>,
    current_yaw_world: f64,
) -> VelocityCmd {
    let dx = goal.x_m - current_pos_world.x;
    let dy = goal.y_m - current_pos_world.y;
    // Wrap yaw error into (−π, π] so the body never picks the long
    // way around a ±π singularity.
    let raw_dyaw = goal.yaw_rad - current_yaw_world;
    let dyaw = (raw_dyaw + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;

    let dist_xy = (dx * dx + dy * dy).sqrt();
    if dist_xy < goal.position_tolerance_m && dyaw.abs() < goal.yaw_tolerance_rad {
        return VelocityCmd::zero();
    }

    let max_v = goal.max_v_m_s.max(1e-6);
    let max_wz = goal.max_wz_rad_s.max(1e-6);
    let t_xy = dist_xy / max_v;
    let t_yaw = dyaw.abs() / max_wz;
    // 50 ms floor: prevents an infinite cmd magnitude when very close
    // to the goal (the tolerance check above usually catches this,
    // but the floor guards against tolerance = 0 configurations).
    let t = t_xy.max(t_yaw).max(0.05);

    let v_x_world = (dx / t).clamp(-max_v, max_v);
    let v_y_world = (dy / t).clamp(-max_v, max_v);
    let wz = (dyaw / t).clamp(-max_wz, max_wz);

    // World → body frame (planar). Same convention as
    // `world_to_body_horizontal` from `mpc_controller.rs`.
    let (s, c) = current_yaw_world.sin_cos();
    let vx_body = c * v_x_world + s * v_y_world;
    let vy_body = -s * v_x_world + c * v_y_world;

    VelocityCmd { vx: vx_body, vy: vy_body, wz }
}

/// Lossy projection of `FullCentroidalMpcSolution` into the SRBD-shaped
/// `MpcSolution` for WBC integration compat. Same idea as
/// `to_compat_mpc_solution` in `centroidal_controller.rs`.
fn to_compat_mpc_solution_full(sol: &FullCentroidalMpcSolution) -> MpcSolution {
    let predicted_body_states: Vec<SrbdState> = sol
        .predicted_states
        .iter()
        .map(|s| SrbdState {
            orientation_rpy: s.base_euler_zyx,
            position: s.base_pos_world,
            angular_velocity: s.angular_velocity_world,
            linear_velocity: s.v_com_world,
        })
        .collect();
    let grfs_first_step = sol.first_input.grfs_world;
    let grfs_all_steps: Vec<[Vector3<f64>; N_FEET]> = sol
        .inputs_all_steps
        .iter()
        .map(|u| u.grfs_world)
        .collect();
    let horizon = grfs_all_steps.len();
    MpcSolution {
        grfs_first_step,
        grfs_all_steps,
        // FullCentroidal mode plans foot positions via joint_q in state,
        // not via the SRBD-style additive Δr offset. Report zeros so
        // downstream readers (compute_mpc_footstep) skip the offset.
        foot_offsets_first_step: [nalgebra::Vector3::zeros(); 4],
        foot_offsets_all_steps: vec![[nalgebra::Vector3::zeros(); 4]; horizon],
        predicted_body_states,
        objective: sol.objective,
        solved: sol.solved,
    }
}

#[cfg(test)]
mod goal_pose_tests {
    use super::*;
    use approx::assert_relative_eq;

    fn goal(x: f64, y: f64, yaw: f64) -> GoalPoseWorld {
        GoalPoseWorld {
            x_m: x,
            y_m: y,
            yaw_rad: yaw,
            max_v_m_s: 0.30,
            max_wz_rad_s: 1.00,
            position_tolerance_m: 0.02,
            yaw_tolerance_rad: 0.05,
        }
    }

    /// At-the-goal: command must be exactly zero so the phase
    /// generator holds in stance (the gait does not wander once the
    /// body has arrived).
    #[test]
    fn velocity_cmd_for_goal_is_zero_inside_tolerance() {
        let cmd = velocity_cmd_for_goal(
            goal(0.0, 0.0, 0.0),
            Vector3::new(0.01, 0.0, 0.0),
            0.0,
        );
        assert_eq!(cmd, VelocityCmd::zero());
    }

    /// Pure forward goal: cmd points in +x_body when current yaw = 0
    /// and the goal is ahead. Magnitude is `dx / t_to_target`; for
    /// a 1 m goal at max_v = 0.3 m/s, that's `1.0 / (1.0/0.3) = 0.3`.
    #[test]
    fn velocity_cmd_for_goal_forward_at_yaw_zero() {
        let cmd = velocity_cmd_for_goal(
            goal(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            0.0,
        );
        assert_relative_eq!(cmd.vx, 0.30, epsilon = 1e-9);
        assert_relative_eq!(cmd.vy, 0.00, epsilon = 1e-9);
        assert_relative_eq!(cmd.wz, 0.00, epsilon = 1e-9);
    }

    /// Body has been pushed laterally off the path back to origin
    /// (goal x=1, y=0). The recovered command must include a
    /// **non-zero negative vy_body** to drag the body back toward
    /// y = 0 while still progressing forward in x.
    #[test]
    fn velocity_cmd_for_goal_pulls_back_after_lateral_push() {
        let cmd = velocity_cmd_for_goal(
            goal(1.0, 0.0, 0.0),
            Vector3::new(0.4, 0.3, 0.0),
            0.0,
        );
        // dx = 0.6, dy = −0.3 → dist_xy ≈ 0.671, t = 0.671 / 0.3 ≈ 2.24 s.
        // v_x_world = 0.6 / 2.24 ≈ 0.268; v_y_world = -0.3 / 2.24 ≈ -0.134.
        // yaw = 0 → body == world.
        assert!(cmd.vx > 0.0, "must still progress forward");
        assert!(cmd.vy < 0.0, "must pull back toward y=0 (got {})", cmd.vy);
        assert_relative_eq!(cmd.vx.hypot(cmd.vy), 0.30, epsilon = 1e-9); // = max_v
    }

    /// World ↔ body frame: with yaw = π/2 (body facing +y_world), a
    /// goal in the +x_world direction must produce a **negative
    /// vy_body** (the body sees the goal to its right).
    #[test]
    fn velocity_cmd_for_goal_rotates_with_body_yaw() {
        let cmd = velocity_cmd_for_goal(
            goal(1.0, 0.0, std::f64::consts::FRAC_PI_2),
            Vector3::new(0.0, 0.0, 0.0),
            std::f64::consts::FRAC_PI_2,
        );
        // World err = (+1, 0). yaw = π/2 → R_body_world rotates by -π/2:
        //   vx_body =  cos(π/2)·1 + sin(π/2)·0 = 0
        //   vy_body = -sin(π/2)·1 + cos(π/2)·0 = -1
        // Normalised to max_v = 0.30.
        assert_relative_eq!(cmd.vx, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cmd.vy, -0.30, epsilon = 1e-9);
    }

    /// Yaw error wraps to (−π, π] so the body never picks the long
    /// way around. Goal at +π facing 0 → error should be −π (not +π).
    #[test]
    fn velocity_cmd_for_goal_yaw_wraps_short_way() {
        // current yaw = +π/2, goal yaw = -π/2  → raw err = -π, wraps to -π (or +π edge).
        // Use yaw goal -3π/4 from +3π/4: raw err = -3π/2, wraps to +π/2.
        let cmd = velocity_cmd_for_goal(
            GoalPoseWorld {
                x_m: 0.0,
                y_m: 0.0,
                yaw_rad: -3.0 * std::f64::consts::FRAC_PI_4,
                max_v_m_s: 0.30,
                max_wz_rad_s: 1.0,
                position_tolerance_m: 0.001,
                yaw_tolerance_rad: 0.001,
            },
            Vector3::new(0.0, 0.0, 0.0),
            3.0 * std::f64::consts::FRAC_PI_4,
        );
        // dyaw = -3π/4 - 3π/4 = -3π/2 → wraps to +π/2 (going the short way).
        assert!(cmd.wz > 0.0, "should pick the short rotation direction (got wz={})", cmd.wz);
    }
}
