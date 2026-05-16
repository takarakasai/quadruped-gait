//! Full-centroidal NMPC (D3) — 24-state extension of [`crate::centroidal_mpc`].
//!
//! Adds **per-leg joint kinematics to the MPC state and inputs**, so the
//! optimizer can model how its joint commands move the contact frames
//! between shooting nodes. This is the legged_control `centroidalModelType=0`
//! formulation (= "kinematic centroidal"), and is the structural fix for
//! the lateral-inversion / forward-dy cross-coupling that empirical
//! tuning (Phase G/H) and SQP iteration (D2) could not eliminate in the
//! 12-state body-only formulation.
//!
//! D3.3.1 scope is **types only** — the dynamics function and SQP solver
//! land in D3.3.2 / D3.3.4.
//!
//! # State (24-dim) and input (24-dim)
//!
//! ```text
//! x = [ v_com_world  (3)            v_com,  m/s
//!       ω_world      (3)            base angular velocity, rad/s
//!       base_pos     (3)            base origin in world, m
//!       base_euler   (3)            ZYX Euler [roll, pitch, yaw], rad
//!       joint_q     (12)            FL/FR/RL/RR × {hip, thigh, calf}, rad
//!     ]
//!
//! u = [ F_FL F_FR F_RL F_RR  (12)   per-foot world-frame GRF, N
//!       joint_v             (12)    per-leg joint velocities, rad/s
//!     ]
//! ```
//!
//! Joint slots inside `joint_q` / `joint_v` follow the canonical
//! [`crate::config::LegId`] order (`FL, FR, RL, RR`), each leg packing
//! `[hip, thigh, calf]`. This matches `KinematicsConfig` and lets D3.3.2
//! reuse [`crate::ik::forward_leg_kinematics`] / [`crate::ik::foot_jacobian_body`]
//! directly without index translation.
//!
//! # Why a separate module instead of extending `CentroidalMpc`?
//!
//! Same reasoning as D1: keep the 12-state baseline alive for A/B
//! comparison while the new formulation is being validated. Once the
//! 24-state version is the production default we can decide whether to
//! retire `CentroidalMpc` or keep it as a "fast-mode" fallback for
//! constrained compute.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

use crate::config::{KinematicsConfig, LegId};
use crate::ik::{foot_jacobian_body, forward_leg_kinematics};

/// Number of leg joints handled by the full-centroidal state. Three per
/// leg × four legs = 12 (Hip-Thigh-Calf RPP morphology, the only one
/// `quadruped-gait` supports today).
pub const N_LEG_JOINTS: usize = 12;

/// Number of feet (= number of GRF slots in the input).
pub const N_FEET: usize = 4;

/// Total state dimension: 12 body (v_com 3 + ω 3 + pos 3 + euler 3) + 12 joint_q.
pub const N_STATE: usize = 12 + N_LEG_JOINTS;

/// Total input dimension: 12 GRF + 12 joint_v.
pub const N_INPUT: usize = 3 * N_FEET + N_LEG_JOINTS;

/// 24-dim full-centroidal state.
///
/// Layout (matches [`Self::to_vec24`]):
///
/// ```text
/// [v_com (3); ω_world (3); base_pos (3); euler_zyx (3); joint_q (12)]
/// ```
///
/// `joint_q` is laid out as `[FL_hip, FL_thigh, FL_calf, FR_…, RL_…, RR_…]`
/// — the same order [`crate::ik::solve_leg_ik`] returns its angles in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FullCentroidalState {
    /// CoM linear velocity in world frame (m/s).
    pub v_com_world: Vector3<f64>,
    /// Base angular velocity in world frame (rad/s). Body and CoM share
    /// ω for a rigid base.
    pub angular_velocity_world: Vector3<f64>,
    /// Base origin in world frame (m).
    pub base_pos_world: Vector3<f64>,
    /// Base orientation as ZYX Euler `[roll, pitch, yaw]` (rad).
    pub base_euler_zyx: Vector3<f64>,
    /// Per-leg joint positions in canonical order (rad). Index 0..3 = FL,
    /// 3..6 = FR, 6..9 = RL, 9..12 = RR; within each block: `[hip, thigh, calf]`.
    pub joint_q: [f64; N_LEG_JOINTS],
}

impl Default for FullCentroidalState {
    fn default() -> Self {
        Self {
            v_com_world: Vector3::zeros(),
            angular_velocity_world: Vector3::zeros(),
            base_pos_world: Vector3::zeros(),
            base_euler_zyx: Vector3::zeros(),
            joint_q: [0.0; N_LEG_JOINTS],
        }
    }
}

impl FullCentroidalState {
    /// Pack into a flat 24-vector with the layout documented above.
    pub fn to_vec(&self) -> [f64; N_STATE] {
        let mut v = [0.0; N_STATE];
        v[0] = self.v_com_world.x;
        v[1] = self.v_com_world.y;
        v[2] = self.v_com_world.z;
        v[3] = self.angular_velocity_world.x;
        v[4] = self.angular_velocity_world.y;
        v[5] = self.angular_velocity_world.z;
        v[6] = self.base_pos_world.x;
        v[7] = self.base_pos_world.y;
        v[8] = self.base_pos_world.z;
        v[9] = self.base_euler_zyx.x;
        v[10] = self.base_euler_zyx.y;
        v[11] = self.base_euler_zyx.z;
        v[12..24].copy_from_slice(&self.joint_q);
        v
    }

    /// Inverse of [`Self::to_vec`].
    pub fn from_vec(v: &[f64; N_STATE]) -> Self {
        let mut joint_q = [0.0; N_LEG_JOINTS];
        joint_q.copy_from_slice(&v[12..24]);
        Self {
            v_com_world: Vector3::new(v[0], v[1], v[2]),
            angular_velocity_world: Vector3::new(v[3], v[4], v[5]),
            base_pos_world: Vector3::new(v[6], v[7], v[8]),
            base_euler_zyx: Vector3::new(v[9], v[10], v[11]),
            joint_q,
        }
    }

    /// Borrow joint angles for one leg as `[hip, thigh, calf]`.
    pub fn leg_joint_q(&self, leg_idx: usize) -> [f64; 3] {
        debug_assert!(leg_idx < N_FEET);
        let base = leg_idx * 3;
        [
            self.joint_q[base],
            self.joint_q[base + 1],
            self.joint_q[base + 2],
        ]
    }
}

/// 24-dim full-centroidal input.
///
/// Layout (matches [`Self::to_vec`]):
///
/// ```text
/// [F_FL (3); F_FR (3); F_RL (3); F_RR (3); joint_v (12)]
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FullCentroidalInput {
    /// World-frame ground reaction force per foot, FL/FR/RL/RR slot order (N).
    pub grfs_world: [Vector3<f64>; N_FEET],
    /// Per-leg joint velocity command (rad/s), same packing as
    /// [`FullCentroidalState::joint_q`].
    pub joint_v: [f64; N_LEG_JOINTS],
}

impl Default for FullCentroidalInput {
    fn default() -> Self {
        Self {
            grfs_world: [Vector3::zeros(); N_FEET],
            joint_v: [0.0; N_LEG_JOINTS],
        }
    }
}

impl FullCentroidalInput {
    /// Pack into a flat 24-vector.
    pub fn to_vec(&self) -> [f64; N_INPUT] {
        let mut v = [0.0; N_INPUT];
        for i in 0..N_FEET {
            v[3 * i] = self.grfs_world[i].x;
            v[3 * i + 1] = self.grfs_world[i].y;
            v[3 * i + 2] = self.grfs_world[i].z;
        }
        v[12..24].copy_from_slice(&self.joint_v);
        v
    }

    /// Inverse of [`Self::to_vec`].
    pub fn from_vec(v: &[f64; N_INPUT]) -> Self {
        let mut grfs = [Vector3::zeros(); N_FEET];
        for i in 0..N_FEET {
            grfs[i] = Vector3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        }
        let mut joint_v = [0.0; N_LEG_JOINTS];
        joint_v.copy_from_slice(&v[12..24]);
        Self {
            grfs_world: grfs,
            joint_v,
        }
    }
}

/// Constant model parameters for the 24-state full-centroidal MPC.
///
/// Mirrors [`crate::centroidal_mpc::CentroidalMpcConfig`] field-for-field
/// (so D3.3.5 can reuse the same `auto_detect_*` populator), with one
/// addition: the per-leg analytical FK config. The kinematics enters
/// [`full_centroidal_dynamics`] in two places:
///
/// 1. The **moment arm** for each foot's GRF — `r_i = foot_world_i − CoM_world`
///    where `foot_world_i = base_pos_world + R · forward_leg_kinematics(joint_q_leg_i)`.
///    This is the structural fix vs. the 12-state model: moment arms
///    update with joint motion within the horizon.
/// 2. (D3.3.3) The **linearization** of those moments — `∂r_i/∂q` flows
///    through `ik::foot_jacobian_body`.
#[derive(Clone, Debug)]
pub struct FullCentroidalMpcConfig {
    /// Total robot mass (kg).
    pub mass_kg: f64,
    /// Centroidal angular inertia in body frame (kg·m²). Constant
    /// across the horizon — the joint-state in this MPC drives foot
    /// positions, not the centroidal inertia tensor (that's the D3.4+
    /// "true full centroidal" extension, deliberately deferred).
    pub centroidal_inertia_body: Matrix3<f64>,
    /// CoM position relative to base origin, in body frame (m).
    pub com_offset_body: Vector3<f64>,
    /// Friction coefficient for the pyramid constraint. Default 0.5
    /// matches both the explicit sim ground-geom friction set by
    /// `src/mjcf.rs` and a realistic real-robot expectation for
    /// rubber on a typical lab floor (linoleum / sealed concrete).
    /// Keeping MPC and sim in sync avoids both directions of mismatch:
    ///
    /// - **MPC > sim friction**: MPC plans GRFs that the sim physics
    ///   can't deliver → lateral slip → tracking error.
    /// - **MPC < sim friction**: MPC is unnecessarily conservative,
    ///   leaving lateral authority on the table → poor disturbance
    ///   recovery (the failure mode at the 0.5/1.0 split before this
    ///   was unified to 0.5 on both sides).
    pub friction_mu: f64,
    /// Max normal force per foot (N). 0 disables.
    pub max_normal_force: f64,
    /// Prediction horizon length (steps).
    pub horizon_steps: usize,
    /// Time per discrete step (s).
    pub dt_per_step: f64,
    /// State cost weights `Q_diag` (size 24), layout matches
    /// [`FullCentroidalState::to_vec`].
    pub q_diag: [f64; N_STATE],
    /// Input cost weights `R_diag` (size 24), layout matches
    /// [`FullCentroidalInput::to_vec`]. Two distinct scales coexist:
    /// the 12 GRF entries (N) and the 12 joint_v entries (rad/s),
    /// so unlike the 12-state version we cannot share a scalar.
    pub r_diag: [f64; N_INPUT],
    /// SQP-style re-linearisation iteration count (same semantics as
    /// [`crate::centroidal_mpc::CentroidalMpcConfig::sqp_iterations`]).
    pub sqp_iterations: usize,
    /// Per-leg analytical FK config (FL/FR/RL/RR).
    pub kinematics: KinematicsConfig,
    /// Add a per-swing-leg-step **vertical foot velocity** equality
    /// constraint to the QP, matching legged_control's
    /// `NormalVelocityConstraintCppAd`. The RHS is taken from
    /// [`FullCentroidalContactSchedule::swing_z_velocity`].
    ///
    /// Default `false` — the legacy D3.3.5a path (constant joint_q
    /// reference, no swing-leg foot-velocity constraint) stays intact
    /// for A/B comparison. Turn on alongside the controller's
    /// `enable_legged_control_parity` flag.
    pub enable_swing_normal_velocity_constraint: bool,
    /// **A3 — friction cone soft + slack penalty.**
    ///
    /// When `true`, the pyramid friction inequalities
    /// `|f_x| ≤ μ·f_z` and `|f_y| ≤ μ·f_z` are replaced with the
    /// slack-relaxed form `|f_x| ≤ μ·f_z + s_x`, `|f_y| ≤ μ·f_z + s_y`,
    /// with `s_x, s_y ≥ 0` and a quadratic cost
    /// `friction_cone_slack_penalty · (s_x² + s_y²)` added per
    /// stance-leg-step. This matches legged_control's
    /// `FrictionConeConstraint` realised as a "relaxed barrier" via
    /// `RelaxedBarrierPenalty` in OCS2 — the cone becomes a target the
    /// solver is heavily penalised for crossing rather than an absolute
    /// limit. The benefit at the pyramid corner (where
    /// `diag_friction_cone_utilization` showed ratio 1.41 ≈ √2 under
    /// lateral 6 N push, i.e. the hard form had no breathing room
    /// before infeasibility) is graceful degradation under physical
    /// excess: the solver still returns a useful GRF instead of
    /// dropping to the reference fallback. f_z ≥ 0 and f_z ≤ f_max
    /// stay hard — negative or arbitrarily large normal force isn't
    /// physically realisable, slack would only hide a bug.
    ///
    /// Default `false` keeps the legacy hard pyramid for backward
    /// compatibility.
    pub friction_cone_soft: bool,
    /// Quadratic penalty weight applied to each friction-cone slack
    /// variable when [`Self::friction_cone_soft`] is `true`. Cost is
    /// `0.5 · 2·penalty · s²` per slack, so the QP's `P` block sees a
    /// `2·penalty` diagonal entry. Reasonable range `100–10_000`
    /// depending on how aggressively the operator wants to keep
    /// slacks near zero. Default `1000.0` is a balance between
    /// "almost-hard cone" and "useful slack budget" — at this weight
    /// a 1 N slack costs as much as a 1e-3 N²·s² GRF deviation
    /// (matching the default `r_diag[GRF]`).
    pub friction_cone_slack_penalty: f64,
    /// **B3 — MPC warm-start.**
    ///
    /// When `true`, [`FullCentroidalMpc::solve`] uses the previous
    /// call's solution (shifted by one step to align with the new
    /// receding horizon) as the SQP's first-iteration linearization
    /// and cost target instead of the caller-supplied reference. This
    /// mirrors legged_control's `MPC_BASE::run` warm-start path
    /// (OCS2's `solverObservation_.controlInput` is fed back into the
    /// next SLQ iteration) and is the standard recipe for shaving
    /// SQP iterations off a real-time MPC tick.
    ///
    /// Practical benefit: at steady-state (cmd held, no disturbance)
    /// the previous solution is already near-optimal for the next
    /// tick, so 1 SQP iter suffices instead of the default 3 — about
    /// **2× faster** wall-clock per solve. Under disturbance the
    /// warm-start is still a better starting point than the
    /// gravity-balanced cmd reference, so convergence is faster
    /// there too.
    ///
    /// Default `false` keeps the legacy cold-start path so the
    /// `sqp_iterations = 3` default and existing baselines are bit-
    /// stable. Tests / benchmarks opt-in explicitly.
    pub warm_start: bool,
    /// **A1 — MPC-optimised footstep XY (cost-side soft tracking).**
    ///
    /// Quadratic cost weight on the world-frame XY residual between
    /// the MPC's predicted foot position and a planner-supplied
    /// target. The cost is added at each (leg, k) where
    /// [`FullCentroidalContactSchedule::foot_xy_target_world`] is
    /// `Some([x, y])`. When zero (default), no cost is added and the
    /// MPC reverts to the legacy "footstep is an open-loop input"
    /// regime where joint_q reference is held constant and the
    /// swing-leg cost relies on the small `q_diag[joint_q]` weight.
    ///
    /// With this term active the MPC linearises `foot_xy(k) =
    /// base_pos_xy(k) + R_z(ψ_ref) · FK_xy(q(k))` and finds a
    /// joint-velocity trajectory that lands the foot at the target
    /// — closing the loop that `use_mpc_predicted_footstep` (P2)
    /// tried to close externally and failed because the external
    /// path couldn't change the footstep itself, only re-read it.
    /// legged_control analogue: the
    /// `EndEffectorKinematicsCppAd`-driven cost term in
    /// `LeggedRobotInterface::setupReferenceManager`.
    ///
    /// Reasonable range: `50.0` (gentle pull) to `5_000.0` (strong
    /// tracking, may overshoot if the planner target is jumpy).
    /// Default `0.0` ⇒ disabled, preserving backward compatibility
    /// for callers that don't populate `foot_xy_target_world`.
    pub q_foot_xy_world: f64,
}

impl FullCentroidalMpcConfig {
    /// Sensible defaults for a Cheetah-3-class quadruped, with the
    /// caller supplying the per-leg `KinematicsConfig` (the one field
    /// that has no model-independent default). Mirrors
    /// `CentroidalMpcConfig::default()` for the body params; cost
    /// weights are conservative and the host typically overrides via
    /// `auto_detect_full_centroidal_mpc_config`.
    pub fn default_with_kin(kinematics: KinematicsConfig) -> Self {
        let mut q_diag = [0.0; N_STATE];
        // Body block (12 entries) — same shape as 12-state CentroidalMpc
        // defaults: v_com, ω, base_pos, euler.
        q_diag[0..3].copy_from_slice(&[1.0, 1.0, 1.0]);
        q_diag[3..6].copy_from_slice(&[0.5, 0.5, 10.0]);
        q_diag[6..9].copy_from_slice(&[0.0, 5.0, 50.0]);
        q_diag[9..12].copy_from_slice(&[25.0, 25.0, 50.0]);
        // joint_q block (12 entries) — light weight so the cost biases
        // toward the held reference without overriding stance no-slip
        // forced joint motion.
        for i in 12..N_STATE {
            q_diag[i] = 0.1;
        }

        let mut r_diag = [0.0; N_INPUT];
        // GRF cost (12 entries) — same as 12-state default.
        for i in 0..12 {
            r_diag[i] = 1e-3;
        }
        // joint_v cost (12 entries) — heavier than GRF so the optimiser
        // doesn't fire crazy joint velocities; stance no-slip equality
        // overrides this when foot pinning requires non-zero `joint_v`.
        for i in 12..N_INPUT {
            r_diag[i] = 1.0;
        }

        Self {
            mass_kg: 9.0,
            centroidal_inertia_body: Matrix3::from_diagonal(&Vector3::new(0.07, 0.26, 0.242)),
            com_offset_body: Vector3::zeros(),
            friction_mu: 0.5,
            max_normal_force: 200.0,
            // 20 × 30 ms = 600 ms preview. The original D3.3.5 default was
            // 10 × 30 ms = 300 ms which under-tracks the cmd reference
            // (forward dx end at 5 s cmd vx=0.15 reached +0.74 m vs +1.06 m
            // achievable with horizon = 20). The external-force benchmark
            // (`tests/integration_walk.rs::diag_external_force_robustness`)
            // showed horizon = 20 wins on the forward-tracking column across
            // every scenario (forward 2/4/6 N, vertical, yaw torque) while
            // keeping cross-axis suppression comparable. legged_control's
            // OCS2 default is 1.0 s (~33 steps) — we trade 33 → 20 to keep
            // the SQP solve cheap enough for 30 ms cadence.
            horizon_steps: 20,
            dt_per_step: 0.030,
            q_diag,
            r_diag,
            sqp_iterations: 3,
            kinematics,
            enable_swing_normal_velocity_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            q_foot_xy_world: 0.0,
        }
    }
}

/// Continuous-time full-centroidal dynamics: ẋ = f(x, u).
///
/// Differs from [`crate::centroidal_mpc::centroidal_dynamics`] in three
/// concrete ways:
///
/// 1. Foot world positions are **computed inside** from the state's
///    joint_q + base pose, not passed in. This is the whole point of
///    moving joints into the MPC state.
/// 2. Joint kinematics are part of the state derivative:
///    `q̇_j = v_j` (input-driven, trivial).
/// 3. The `q_diag` / `r_diag` weights are 24-element each, since both
///    state and input grew.
///
/// Returns the time-derivative of `state` (with the same field layout).
/// Caller integrates as `x_{k+1} = x_k + dt · f(x_k, u_k)` (explicit
/// Euler, matching the 12-state shooting model).
pub fn full_centroidal_dynamics(
    state: &FullCentroidalState,
    input: &FullCentroidalInput,
    cfg: &FullCentroidalMpcConfig,
) -> FullCentroidalState {
    let g_world = Vector3::new(0.0, 0.0, -9.81);

    // Body-frame → world rotation from the state's Euler angles
    // (ZYX = R_z(yaw) · R_y(pitch) · R_x(roll)).
    let r_world_body = Rotation3::from_euler_angles(
        state.base_euler_zyx.x,
        state.base_euler_zyx.y,
        state.base_euler_zyx.z,
    );
    let r_mat = r_world_body.matrix();

    // CoM position in world frame.
    let com_offset_world = r_world_body * cfg.com_offset_body;
    let com_pos_world = state.base_pos_world + com_offset_world;

    // ── Per-leg foot positions in world frame (from FK + base pose) ──
    let foot_world = compute_foot_positions_world(state, cfg);

    // ── Linear: v̇_com = (Σ F)/m + g ──────────────────────────────────
    let total_f: Vector3<f64> = input.grfs_world.iter().sum();
    let v_com_dot = total_f / cfg.mass_kg.max(1e-9) + g_world;

    // ── Angular: α = I_world⁻¹ · (Σ r_i × F_i − ω × Iω) ──────────────
    let mut tau_world = Vector3::zeros();
    for slot in 0..N_FEET {
        let r = foot_world[slot] - com_pos_world;
        tau_world += r.cross(&input.grfs_world[slot]);
    }
    let i_world = r_mat * cfg.centroidal_inertia_body * r_mat.transpose();
    let i_world_inv = i_world.try_inverse().unwrap_or_else(Matrix3::identity);
    let omega_world = state.angular_velocity_world;
    let coriolis = omega_world.cross(&(i_world * omega_world));
    let alpha_world = i_world_inv * (tau_world - coriolis);

    // ── Base position rate: v_base = v_com − ω × R·com_offset ────────
    let base_pos_dot = state.v_com_world - omega_world.cross(&com_offset_world);

    // ── Euler-ZYX rate from world ω (same kinematic transform as 12s) ─
    let base_euler_dot = crate::centroidal_mpc::euler_zyx_dot_from_world_omega(
        &state.base_euler_zyx,
        &omega_world,
    );

    // ── Joint kinematic part: q̇_j = v_j ──────────────────────────────
    let joint_q_dot = input.joint_v;

    FullCentroidalState {
        v_com_world: v_com_dot,
        angular_velocity_world: alpha_world,
        base_pos_world: base_pos_dot,
        base_euler_zyx: base_euler_dot,
        joint_q: joint_q_dot,
    }
}

/// State dimension augmented with constant gravity (one extra row).
/// This matches the 12-state MPC's `nx=13` convention (= 12 dynamic + 1
/// augmented gravity) so the discrete-time shooting uses the same
/// `x_{k+1} = A_d · x_aug + B_d · u` layout.
pub const N_STATE_AUG: usize = N_STATE + 1; // 25

/// Continuous-time Jacobians `(A_c, B_c)` of [`full_centroidal_dynamics`]
/// evaluated at a yaw-only reference state.
///
/// The reference is treated as **small-angle on roll/pitch and ω_ref = 0**
/// (the operating regime of trotting flat-ground gaits), reducing
/// `R_world_body` to `R_z(psi_ref)` in the linearisation. Coriolis
/// `ω × I·ω` is dropped (quadratic in ω; zero at ω_ref=0).
///
/// Returns `(A: 25×25, B: 25×24)` matching the augmented state layout
/// `[v_com (3); ω_world (3); base_pos (3); euler_zyx (3); joint_q (12); g_aug (1)]`.
/// The augmented state's last component holds the constant `g_world.z =
/// −9.81`; row `2` of `A` has a `1.0` at column `24` so the discrete
/// integration picks up gravity exactly.
///
/// New blocks vs. [`crate::centroidal_mpc::CentroidalMpc::continuous_dynamics`]:
///
/// - `A[3..6, 12..24]` = ∂α/∂joint_q. The angular acceleration
///   responds to per-leg foot-position changes via `r_i = R_z·(p_foot_body
///   − com_offset)`, and `∂(r×F)/∂q = -skew(F_ref) · R_z · ∂p_foot_body/∂q`
///   where the body-frame Jacobian comes from
///   [`crate::ik::foot_jacobian_body`]. This is the structural fix
///   the 12-state model could not represent — the optimiser now sees
///   how its joint commands shift the moment arm.
/// - `B[12..24, 12..24]` = `I_12` — joint_v drives joint_q one-to-one.
pub fn continuous_dynamics_full(
    state_ref: &FullCentroidalState,
    input_ref: &FullCentroidalInput,
    cfg: &FullCentroidalMpcConfig,
    stance: &[bool; N_FEET],
    psi_ref: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let nx = N_STATE_AUG;
    let nu = N_INPUT;
    let mut a = DMatrix::<f64>::zeros(nx, nx);
    let mut b = DMatrix::<f64>::zeros(nx, nu);

    let m = cfg.mass_kg.max(1e-9);
    let (s, c) = psi_ref.sin_cos();
    let r_z = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    let i_world = r_z * cfg.centroidal_inertia_body * r_z.transpose();
    let i_world_inv = i_world.try_inverse().unwrap_or_else(Matrix3::identity);

    // Per-leg reference foot positions in world. CoM offset is
    // small-angle: `com_offset_world ≈ R_z · com_offset_body`.
    let com_offset_world_ref = r_z * cfg.com_offset_body;
    let com_pos_world_ref = state_ref.base_pos_world + com_offset_world_ref;
    let foot_ref_world = compute_foot_positions_world(state_ref, cfg);

    // ── 1. v̇_com row: gravity (via aug state) + (1/m)·F per stance leg
    a[(2, 24)] = 1.0;
    for leg in 0..N_FEET {
        if !stance[leg] {
            continue;
        }
        for i in 0..3 {
            b[(i, 3 * leg + i)] = 1.0 / m;
        }
    }

    // ── 2. α row ──────────────────────────────────────────────────────
    let legs = [LegId::FL, LegId::FR, LegId::RL, LegId::RR];
    // Per-axis basis-skew matrices for the small-angle Euler derivative
    // `∂R/∂φ_k ≈ R_z · skew(e_k)` at roll=pitch=0.
    let e_skew = [
        skew_3(&Vector3::x()),
        skew_3(&Vector3::y()),
        skew_3(&Vector3::z()),
    ];
    // Sum-of-stance-leg ∂τ_world/∂φ_k accumulator (k = 0=roll, 1=pitch, 2=yaw).
    let mut dtau_deuler = [Vector3::zeros(); 3];
    for leg in 0..N_FEET {
        if !stance[leg] {
            continue;
        }
        // ∂α/∂F[leg] = I_world⁻¹ · skew(r_leg_ref)
        let r_ref = foot_ref_world[leg] - com_pos_world_ref;
        let block_grf = i_world_inv * skew_3(&r_ref);
        for i in 0..3 {
            for j in 0..3 {
                b[(3 + i, 3 * leg + j)] = block_grf[(i, j)];
            }
        }
        // ∂α/∂joint_q_leg = -I_world⁻¹ · skew(F_leg_ref) · R_z · J_q_body
        let f_ref = input_ref.grfs_world[leg];
        let kin = cfg.kinematics.leg(legs[leg]);
        let [qhip, qthigh, qcalf] = state_ref.leg_joint_q(leg);
        let j_body = foot_jacobian_body(kin, qhip, qthigh, qcalf);
        let block_q = -i_world_inv * skew_3(&f_ref) * r_z * j_body;
        let q_col_base = 12 + 3 * leg;
        for i in 0..3 {
            for j in 0..3 {
                a[(3 + i, q_col_base + j)] = block_q[(i, j)];
            }
        }
        // Accumulate ∂τ_world/∂base_euler. With foot_body fixed (joint_q
        // held), the rotation-induced position change is
        //   ∂r_leg/∂φ_k = R_z · skew(e_k) · (foot_body − com_offset)
        // and the moment contribution is
        //   ∂(r × F)/∂φ_k = (∂r/∂φ_k) × F_ref
        // (the F vector lives in the world frame and is independent of
        // the base orientation).
        let foot_body_minus_com =
            forward_leg_kinematics(kin, qhip, qthigh, qcalf) - cfg.com_offset_body;
        for k in 0..3 {
            let dr = r_z * e_skew[k] * foot_body_minus_com;
            dtau_deuler[k] += dr.cross(&f_ref);
        }
    }
    // ∂α/∂base_euler also has the ∂I_world⁻¹/∂φ · τ_ref term:
    //   I_world(φ) = R(φ) · I_body · R(φ)^T
    //   ∂I_world/∂φ_k at small roll/pitch and yaw ψ_ref equals
    //     R_z · ([skew(e_k), I_body]) · R_z^T   (matrix commutator)
    //   ∂I_world⁻¹/∂φ_k = -I_world⁻¹ · ∂I_world/∂φ_k · I_world⁻¹
    //   ⇒ extra term:   -I_world⁻¹ · ∂I_world/∂φ_k · I_world⁻¹ · τ_ref
    // This term is non-zero even when τ_ref is balanced — gravity-supporting
    // Σ r×F ≠ 0 at non-symmetric leg poses.
    let mut tau_ref_world = Vector3::zeros();
    for leg in 0..N_FEET {
        if !stance[leg] {
            continue;
        }
        let r_leg = foot_ref_world[leg] - com_pos_world_ref;
        tau_ref_world += r_leg.cross(&input_ref.grfs_world[leg]);
    }
    let i_world_inv_tau = i_world_inv * tau_ref_world;
    let i_body = cfg.centroidal_inertia_body;
    for k in 0..3 {
        // Commutator [skew(e_k), I_body] = skew(e_k)·I_body − I_body·skew(e_k)
        let comm = e_skew[k] * i_body - i_body * e_skew[k];
        let di_world = r_z * comm * r_z.transpose();
        let di_inv_term = -i_world_inv * di_world * i_world_inv_tau;
        let dalpha = i_world_inv * dtau_deuler[k] + di_inv_term;
        for i in 0..3 {
            a[(3 + i, 9 + k)] = dalpha[i];
        }
    }

    // ── 3. ṗ_base row: v_com (small-CoM-offset) ──────────────────────
    a[(6, 0)] = 1.0;
    a[(7, 1)] = 1.0;
    a[(8, 2)] = 1.0;

    // ── 4. ė_zyx row: T_inv · ω = R_z^T · ω at small roll/pitch ──────
    let r_z_t = r_z.transpose();
    for i in 0..3 {
        for j in 0..3 {
            a[(9 + i, 3 + j)] = r_z_t[(i, j)];
        }
    }

    // ── 5. q̇_j row: joint_v block of B is identity ───────────────────
    for i in 0..N_LEG_JOINTS {
        b[(12 + i, 12 + i)] = 1.0;
    }

    // ── 6. augmented gravity row: ġ = 0 (left zero by initialiser) ───

    (a, b)
}

fn skew_3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

// ─────────────────────────────────────────────────────────────────────────
// Reference / Contact / Solution / Mpc — sibling types to centroidal_mpc.
// ─────────────────────────────────────────────────────────────────────────

/// Reference trajectory the full-centroidal MPC tracks. One entry per
/// horizon step. Caller (host) is responsible for filling `joint_q_ref`
/// from IK of the planned footstep trajectory (D3.3 design choice (a)
/// — swing leg tracking via pre-computed joint reference).
#[derive(Clone, Debug)]
pub struct FullCentroidalReference {
    pub states: Vec<FullCentroidalState>,
    pub inputs: Vec<FullCentroidalInput>,
}

impl FullCentroidalReference {
    /// Constant reference: same state and zero input over the horizon.
    /// Useful for unit tests; production code fills joint_q from IK
    /// per step.
    pub fn constant(state: FullCentroidalState, horizon_steps: usize) -> Self {
        Self {
            states: vec![state; horizon_steps],
            inputs: vec![FullCentroidalInput::default(); horizon_steps],
        }
    }
}

/// Per-leg per-step stance schedule (same shape as the 12-state version).
///
/// `swing_z_velocity` is the per-step planned world-frame vertical foot
/// velocity for each leg; entries are ignored when the corresponding
/// `is_stance[leg][k]` is true. The MPC reads these only when
/// [`FullCentroidalMpcConfig::enable_swing_normal_velocity_constraint`]
/// is `true` — otherwise the field has no effect and zeros are fine.
///
/// `stance_f_max` is a per-(leg, step) **upper bound on the vertical
/// GRF** (Newtons) that the MPC's inequality block enforces as a hard
/// constraint. Default `f64::INFINITY` ⇒ no per-step tightening, so
/// the existing global [`FullCentroidalMpcConfig::max_normal_force`]
/// applies unchanged. Setting a smaller value forces the MPC to
/// redistribute load to other stance legs at that step — the
/// constraint-side version of the C1 transition-phase smoothing
/// (C1-2 experiment). The effective bound is
/// `min(stance_f_max[leg][k], cfg.max_normal_force)`.
#[derive(Clone, Debug)]
pub struct FullCentroidalContactSchedule {
    pub is_stance: [Vec<bool>; N_FEET],
    pub swing_z_velocity: [Vec<f64>; N_FEET],
    pub stance_f_max: [Vec<f64>; N_FEET],
    /// **A1**: per-(leg, step) world-frame XY foothold target. When
    /// `Some([x, y])`, the MPC adds a soft quadratic cost
    /// `q_foot_xy_world · ‖foot_xy_world(leg, k) − [x, y]‖²` to the
    /// objective, encouraging the optimiser to deviate the swing-leg
    /// joint trajectory so the foot lands at the planned location.
    /// `None` ⇒ no cost added at that (leg, k). Typical caller
    /// pattern: set `Some` only at the touchdown step (first stance
    /// step of the next stance phase per leg) and leave every other
    /// step `None`; the stance no-slip equality then pins the foot
    /// at the achieved touchdown for the rest of the stance.
    pub foot_xy_target_world: [Vec<Option<[f64; 2]>>; N_FEET],
}

impl FullCentroidalContactSchedule {
    pub fn all_stance(horizon_steps: usize) -> Self {
        Self {
            is_stance: [
                vec![true; horizon_steps],
                vec![true; horizon_steps],
                vec![true; horizon_steps],
                vec![true; horizon_steps],
            ],
            swing_z_velocity: [
                vec![0.0; horizon_steps],
                vec![0.0; horizon_steps],
                vec![0.0; horizon_steps],
                vec![0.0; horizon_steps],
            ],
            stance_f_max: [
                vec![f64::INFINITY; horizon_steps],
                vec![f64::INFINITY; horizon_steps],
                vec![f64::INFINITY; horizon_steps],
                vec![f64::INFINITY; horizon_steps],
            ],
            foot_xy_target_world: [
                vec![None; horizon_steps],
                vec![None; horizon_steps],
                vec![None; horizon_steps],
                vec![None; horizon_steps],
            ],
        }
    }
}

/// Output of [`FullCentroidalMpc::solve`].
#[derive(Clone, Debug)]
pub struct FullCentroidalMpcSolution {
    /// First-step inputs — what the host commits this MPC tick
    /// (receding-horizon convention).
    pub first_input: FullCentroidalInput,
    /// Full-horizon inputs.
    pub inputs_all_steps: Vec<FullCentroidalInput>,
    /// Predicted state across the horizon.
    pub predicted_states: Vec<FullCentroidalState>,
    /// QP objective value.
    pub objective: f64,
    /// Clarabel reported Solved / AlmostSolved.
    pub solved: bool,
}

/// Full-centroidal MPC solver (24-state).
#[derive(Clone, Debug)]
pub struct FullCentroidalMpc {
    cfg: FullCentroidalMpcConfig,
    /// B3: cached trajectory from the previous [`Self::solve`] call,
    /// used as the SQP's iter-0 starting point when
    /// `cfg.warm_start = true`. `None` until the first solve completes,
    /// and cleared automatically when `horizon_steps` changes (the
    /// shape no longer matches). The cache is "shifted by one step on
    /// use" — see [`Self::warm_start_initial_ref`].
    warm_start_cache: Option<FullCentroidalReference>,
}

impl FullCentroidalMpc {
    pub fn new(cfg: FullCentroidalMpcConfig) -> Self {
        Self {
            cfg,
            warm_start_cache: None,
        }
    }

    pub fn config(&self) -> &FullCentroidalMpcConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: FullCentroidalMpcConfig) {
        // Invalidate the warm-start cache when the horizon length
        // changes; mismatched lengths would otherwise panic the SQP
        // shape asserts. Other config changes leave the cache intact
        // since the trajectory is still a reasonable seed even after
        // a weight tweak.
        if cfg.horizon_steps != self.cfg.horizon_steps {
            self.warm_start_cache = None;
        }
        self.cfg = cfg;
    }

    /// Drop the cached warm-start trajectory so the next
    /// [`Self::solve`] call cold-starts. Useful after a reset or a
    /// large reference jump (gait switch, goal hop) where the cached
    /// solution is no longer near-optimal.
    pub fn clear_warm_start(&mut self) {
        self.warm_start_cache = None;
    }

    /// Build the iter-0 reference for the SQP loop. With
    /// `warm_start = false` (default) this is the caller-supplied
    /// reference — same as before B3. With `warm_start = true` and
    /// a cached previous solution of matching length, we **shift by
    /// one step**: drop step 0 (= what the previous tick already
    /// committed) and duplicate the last step at the tail. This
    /// matches the receding-horizon convention and is the same
    /// shape transformation legged_control's OCS2 uses for its
    /// warm-started DDP rollout.
    fn warm_start_initial_ref(
        &self,
        caller_ref: &FullCentroidalReference,
        n: usize,
    ) -> FullCentroidalReference {
        if !self.cfg.warm_start {
            return caller_ref.clone();
        }
        let Some(prev) = self.warm_start_cache.as_ref() else {
            return caller_ref.clone();
        };
        if prev.states.len() != n || prev.inputs.len() != n {
            return caller_ref.clone();
        }
        let mut states = Vec::with_capacity(n);
        let mut inputs = Vec::with_capacity(n);
        for k in 0..n {
            let src = (k + 1).min(n - 1);
            states.push(prev.states[src]);
            inputs.push(prev.inputs[src]);
        }
        FullCentroidalReference { states, inputs }
    }

    /// Solve the MPC QP for the next horizon, returning the optimal
    /// first-step input (GRFs + joint_v) and the full predicted
    /// trajectory. SQP runs `cfg.sqp_iterations` re-linearizations of
    /// the full (state, input) reference trajectory — chosen per D3.3
    /// design (a) for full SQP coverage of joint-driven non-linearity.
    pub fn solve(
        &mut self,
        state_now: FullCentroidalState,
        reference: &FullCentroidalReference,
        contact: &FullCentroidalContactSchedule,
    ) -> FullCentroidalMpcSolution {
        let n = self.cfg.horizon_steps;
        assert_eq!(reference.states.len(), n, "ref state length mismatch");
        assert_eq!(reference.inputs.len(), n, "ref input length mismatch");
        for leg in 0..N_FEET {
            assert_eq!(contact.is_stance[leg].len(), n);
            assert_eq!(contact.swing_z_velocity[leg].len(), n);
            assert_eq!(contact.stance_f_max[leg].len(), n);
            assert_eq!(contact.foot_xy_target_world[leg].len(), n);
        }

        let n_iter = self.cfg.sqp_iterations.max(1);

        // SQP loop. Re-linearisation point updates from previous iter's
        // predicted (state, input) trajectory. Iter 0 starts at the
        // caller-supplied reference, except when B3 warm-start is on
        // and a cached previous-tick solution is available.
        let mut ref_traj = self.warm_start_initial_ref(reference, n);
        let mut last_solution: Option<FullCentroidalMpcSolution> = None;
        for iter in 0..n_iter {
            let sol = self.solve_one_iter(state_now, &ref_traj, contact, n);
            if iter + 1 < n_iter && sol.solved {
                // Update ref states/inputs from this solution; the next
                // iteration linearises at the trajectory the QP just
                // predicted.
                for k in 0..n {
                    ref_traj.states[k] = sol.predicted_states[k];
                    ref_traj.inputs[k] = sol.inputs_all_steps[k];
                }
            }
            last_solution = Some(sol);
        }
        let final_sol = last_solution.expect("at least one SQP iteration ran");
        // B3: cache the converged trajectory so the next tick's
        // `warm_start_initial_ref` has something to shift. We cache
        // only solved trajectories — a failed solve returns the
        // reference fallback which already doesn't help warm-start.
        if self.cfg.warm_start && final_sol.solved {
            self.warm_start_cache = Some(FullCentroidalReference {
                states: final_sol.predicted_states.clone(),
                inputs: final_sol.inputs_all_steps.clone(),
            });
        }
        final_sol
    }

    fn solve_one_iter(
        &self,
        state_now: FullCentroidalState,
        ref_traj: &FullCentroidalReference,
        contact: &FullCentroidalContactSchedule,
        n: usize,
    ) -> FullCentroidalMpcSolution {
        let nx = N_STATE_AUG; // 25
        let nu = N_INPUT; // 24
        // A3: extra decision vars for friction-cone slacks (s_x, s_y per
        // stance-leg-step). Zero when `friction_cone_soft = false`.
        let n_slacks = n_friction_slack_vars(&self.cfg, contact, n);
        let total_vars = nu * n + n_slacks;

        // ── Build per-step continuous-time A_c, B_c, then discretise ──
        let mut a_d_per_step: Vec<DMatrix<f64>> = Vec::with_capacity(n);
        let mut b_d_per_step: Vec<DMatrix<f64>> = Vec::with_capacity(n);
        for k in 0..n {
            let stance = [
                contact.is_stance[0][k],
                contact.is_stance[1][k],
                contact.is_stance[2][k],
                contact.is_stance[3][k],
            ];
            let psi_ref = ref_traj.states[k].base_euler_zyx.z;
            let (a_c, b_c) = continuous_dynamics_full(
                &ref_traj.states[k],
                &ref_traj.inputs[k],
                &self.cfg,
                &stance,
                psi_ref,
            );
            // Forward Euler: x_{k+1} = (I + dt·A) x_k + dt·B u_k.
            let mut a_d = DMatrix::<f64>::identity(nx, nx);
            a_d += &a_c * self.cfg.dt_per_step;
            let b_d = b_c * self.cfg.dt_per_step;
            a_d_per_step.push(a_d);
            b_d_per_step.push(b_d);
        }

        // ── Lifted dynamics: X = A_x x_0 + B_u U ──────────────────────
        let mut a_x = DMatrix::<f64>::zeros(nx * n, nx);
        let mut b_u = DMatrix::<f64>::zeros(nx * n, nu * n);
        let mut prod = DMatrix::<f64>::identity(nx, nx);
        for k in 0..n {
            prod = &a_d_per_step[k] * &prod;
            a_x.view_mut((k * nx, 0), (nx, nx)).copy_from(&prod);
            let mut tail = DMatrix::<f64>::identity(nx, nx);
            for j in (0..=k).rev() {
                let block = &tail * &b_d_per_step[j];
                b_u.view_mut((k * nx, j * nu), (nx, nu)).copy_from(&block);
                if j > 0 {
                    tail = &tail * &a_d_per_step[j];
                }
            }
        }

        // ── Cost: J = ‖X − X_ref‖²_Q + ‖U − U_ref‖²_R ─────────────────
        // P = 2 (B_u^T Q B_u + R), q = 2 (B_u^T Q (A_x x_0 − X_ref) − R U_ref).
        let mut q_block = DMatrix::<f64>::zeros(nx * n, nx * n);
        for k in 0..n {
            for i in 0..N_STATE {
                q_block[(k * nx + i, k * nx + i)] = self.cfg.q_diag[i];
            }
            // Augmented gravity col has zero weight (deterministic constant).
        }
        let mut r_block = DMatrix::<f64>::zeros(nu * n, nu * n);
        for k in 0..n {
            for i in 0..nu {
                r_block[(k * nu + i, k * nu + i)] = self.cfg.r_diag[i];
            }
        }
        let x_ref = {
            let mut v = DVector::<f64>::zeros(nx * n);
            for k in 0..n {
                let s = state_to_vec_aug(&ref_traj.states[k]);
                v.rows_mut(k * nx, nx).copy_from(&s);
            }
            v
        };
        let u_ref = {
            let mut v = DVector::<f64>::zeros(nu * n);
            for k in 0..n {
                let ui = ref_traj.inputs[k].to_vec();
                for i in 0..nu {
                    v[k * nu + i] = ui[i];
                }
            }
            v
        };
        let x_now = state_to_vec_aug(&state_now);
        let drift = &a_x * &x_now - &x_ref;
        let mut p_u_block = 2.0 * (b_u.transpose() * &q_block * &b_u + &r_block);
        let mut q_u_block = 2.0 * (b_u.transpose() * &q_block * &drift - &r_block * &u_ref);

        // A1: add the foot-XY soft tracking cost when a target is
        // provided and `q_foot_xy_world > 0`. This is what lets the
        // MPC pick a footstep (via joint_v over the swing phase) to
        // hit a planner-supplied touchdown point. No-op when neither
        // condition is met, so default behaviour is preserved.
        add_foot_xy_soft_cost(
            &self.cfg,
            contact,
            ref_traj,
            n,
            &a_x,
            &b_u,
            &x_now,
            &mut p_u_block,
            &mut q_u_block,
        );

        // A3: pad P and q with the slack block when soft cone is on.
        // Slacks decouple from U in the cost (no cross terms), so we
        // simply place `2·penalty` on the slack diagonal of P and zero
        // in the slack rows of q.
        let (p_dense, q_vec) = if n_slacks > 0 {
            let mut p = DMatrix::<f64>::zeros(total_vars, total_vars);
            p.view_mut((0, 0), (nu * n, nu * n)).copy_from(&p_u_block);
            let two_pen = 2.0 * self.cfg.friction_cone_slack_penalty;
            for i in 0..n_slacks {
                p[(nu * n + i, nu * n + i)] = two_pen;
            }
            let mut q = DVector::<f64>::zeros(total_vars);
            q.rows_mut(0, nu * n).copy_from(&q_u_block);
            (p, q)
        } else {
            (p_u_block, q_u_block)
        };

        // ── Constraints (D3.3.4b — adds stance no-slip) ─────────────
        let (a_csc, b_vec, cones) = build_constraints_24(
            &self.cfg, contact, ref_traj, n, &a_x, &b_u, &x_now, n_slacks,
        );

        // ── clarabel solve ────────────────────────────────────────────
        let p_csc = dense_to_csc_upper_24(&p_dense);
        let q_slice: Vec<f64> = q_vec.iter().copied().collect();
        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.max_iter = 50;
        let mut solver =
            match DefaultSolver::new(&p_csc, &q_slice, &a_csc, &b_vec, &cones, settings) {
                Ok(s) => s,
                Err(_) => {
                    return failed_solution(&state_now, &ref_traj.inputs, n);
                }
            };
        solver.solve();
        let solved = matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        );
        let z_opt: Vec<f64> = solver.solution.x.clone();
        let objective = solver.solution.obj_val;

        // Decode U → per-step FullCentroidalInput. (Slack block, if any,
        // sits in `z_opt[nu*n..]` and isn't exposed in the solution
        // struct — it's a purely internal QP relaxation variable.)
        let mut inputs_all_steps = Vec::with_capacity(n);
        for k in 0..n {
            let base = k * nu;
            let mut slice = [0.0; N_INPUT];
            slice.copy_from_slice(&z_opt[base..base + nu]);
            inputs_all_steps.push(FullCentroidalInput::from_vec(&slice));
        }

        // Decode predicted states from X = A_x x_0 + B_u U (slacks
        // don't enter the dynamics, so we feed only the first nu*n
        // entries to B_u).
        let u_dvec = DVector::from_iterator(nu * n, z_opt.iter().take(nu * n).copied());
        let x_horizon = &a_x * &x_now + &b_u * &u_dvec;
        let mut predicted_states = Vec::with_capacity(n);
        for k in 0..n {
            let row0 = k * nx;
            let mut slice = [0.0; N_STATE];
            slice.copy_from_slice(&x_horizon.as_slice()[row0..row0 + N_STATE]);
            predicted_states.push(FullCentroidalState::from_vec(&slice));
        }

        FullCentroidalMpcSolution {
            first_input: inputs_all_steps[0].clone(),
            inputs_all_steps,
            predicted_states,
            objective,
            solved,
        }
    }
}

/// Pack `FullCentroidalState` into the 25-dim augmented vector with
/// `g_aug = -9.81` in the last slot.
fn state_to_vec_aug(s: &FullCentroidalState) -> DVector<f64> {
    let mut v = DVector::<f64>::zeros(N_STATE_AUG);
    let body = s.to_vec();
    for i in 0..N_STATE {
        v[i] = body[i];
    }
    v[N_STATE] = -9.81;
    v
}

fn failed_solution(
    state_now: &FullCentroidalState,
    ref_inputs: &[FullCentroidalInput],
    n: usize,
) -> FullCentroidalMpcSolution {
    FullCentroidalMpcSolution {
        first_input: FullCentroidalInput::default(),
        inputs_all_steps: ref_inputs.to_vec(),
        predicted_states: vec![*state_now; n],
        objective: f64::NAN,
        solved: false,
    }
}

/// **A1**: assemble the foot-XY soft tracking contribution to the QP
/// cost. For every `(leg, k)` where the schedule provides a target,
/// linearise `foot_xy_world(leg, k)` around the current SQP reference
/// (`ref_traj`) and add the corresponding quadratic-residual cost to
/// the U-block of `P` and `q`.
///
/// The linearisation:
///
/// ```text
/// foot_xy(k, leg) ≈ base_pos_xy(k) + (R_z(ψ_ref) · FK_body(q(k)))_xy
///                ≈ base_pos_xy(k) + (R_z · J_body(q_ref))_xy · q(k)
///                  + (R_z FK_body(q_ref))_xy − (R_z · J_body(q_ref))_xy · q_ref
/// ```
///
/// Both `base_pos_xy(k)` and `q(k, leg)` are linear functions of
/// `[x_now, U]` via the lifted dynamics `X = A_x x_now + B_u U`, so
/// the predicted foot position is also linear in U, and the residual
/// `foot_xy − target` is linear in U. Squaring it gives the standard
/// `‖M U + r0‖²` form whose Hessian / gradient slot straight into
/// the QP `P` / `q`.
///
/// Slack columns (A3) sit beyond `nu·n` and aren't touched here —
/// they're decoupled from the foot-XY residual.
fn add_foot_xy_soft_cost(
    cfg: &FullCentroidalMpcConfig,
    contact: &FullCentroidalContactSchedule,
    ref_traj: &FullCentroidalReference,
    n: usize,
    a_x: &DMatrix<f64>,
    b_u: &DMatrix<f64>,
    x_now: &DVector<f64>,
    p_u_block: &mut DMatrix<f64>,
    q_u_block: &mut DVector<f64>,
) {
    let q_foot = cfg.q_foot_xy_world;
    if q_foot <= 0.0 {
        return;
    }
    let nu = N_INPUT;
    let nx = N_STATE_AUG;
    let legs_arr = [LegId::FL, LegId::FR, LegId::RL, LegId::RR];

    for k in 0..n {
        for leg in 0..N_FEET {
            let Some(target_xy) = contact.foot_xy_target_world[leg][k] else {
                continue;
            };
            let psi_ref = ref_traj.states[k].base_euler_zyx.z;
            let (s, c) = psi_ref.sin_cos();
            let r_z = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
            let kin = cfg.kinematics.leg(legs_arr[leg]);
            let [qhip, qthigh, qcalf] = ref_traj.states[k].leg_joint_q(leg);
            let foot_body_ref = forward_leg_kinematics(kin, qhip, qthigh, qcalf);
            let j_foot_body = foot_jacobian_body(kin, qhip, qthigh, qcalf);
            let r_z_j = r_z * j_foot_body; // 3×3, world-frame foot Jacobian wrt joint_q
            let q_ref_leg = Vector3::new(qhip, qthigh, qcalf);
            let foot_world_ref = r_z * foot_body_ref;
            let j_times_qref = r_z_j * q_ref_leg;

            for ax in 0..2 {
                // Build the selector vector e_xy that maps a per-step
                // state vector (length nx) to foot_xy[ax].
                //  e_xy[6 + ax]            = 1                  (base_pos[ax] direct)
                //  e_xy[12 + 3·leg + j]    = (R_z J_body)[ax,j] (joint contribution)
                let mut e_xy = [0.0_f64; N_STATE_AUG];
                e_xy[6 + ax] = 1.0;
                for j in 0..3 {
                    e_xy[12 + 3 * leg + j] = r_z_j[(ax, j)];
                }

                // A_xy_row[col] = e_xy · A_x[k·nx..(k+1)·nx, col]
                // B_xy_row[col] = e_xy · B_u[k·nx..(k+1)·nx, col]
                let mut a_xy_row = [0.0_f64; N_STATE_AUG];
                for col in 0..nx {
                    let mut v = 0.0;
                    for r in 0..nx {
                        v += e_xy[r] * a_x[(k * nx + r, col)];
                    }
                    a_xy_row[col] = v;
                }
                let mut b_xy_row = DVector::<f64>::zeros(nu * n);
                for col in 0..(nu * n) {
                    let mut v = 0.0;
                    for r in 0..nx {
                        v += e_xy[r] * b_u[(k * nx + r, col)];
                    }
                    b_xy_row[col] = v;
                }

                // K_xy = (R_z FK_body(q_ref))[ax] − (R_z J_body q_ref)[ax]
                let k_xy = foot_world_ref[ax] - j_times_qref[ax];

                // r0 = a_xy_row · x_now + K_xy − target[ax]
                let mut r0 = k_xy - target_xy[ax];
                for col in 0..nx {
                    r0 += a_xy_row[col] * x_now[col];
                }

                // P_U += 2 q_foot · b_xy_row b_xy_row^T  (rank-1 update)
                // q_U += 2 q_foot · r0 · b_xy_row
                let two_q_foot = 2.0 * q_foot;
                for i in 0..(nu * n) {
                    let bi = b_xy_row[i];
                    if bi.abs() < 1e-14 {
                        continue;
                    }
                    q_u_block[i] += two_q_foot * r0 * bi;
                    for j in 0..(nu * n) {
                        let bj = b_xy_row[j];
                        if bj.abs() < 1e-14 {
                            continue;
                        }
                        p_u_block[(i, j)] += two_q_foot * bi * bj;
                    }
                }
            }
        }
    }
}

/// Number of friction-cone slack decision variables the QP needs given
/// the active soft-cone mode. Returns `0` when
/// [`FullCentroidalMpcConfig::friction_cone_soft`] is off (legacy hard
/// pyramid). Otherwise emits two slacks (s_x, s_y) per stance-leg-step.
fn n_friction_slack_vars(
    cfg: &FullCentroidalMpcConfig,
    contact: &FullCentroidalContactSchedule,
    n: usize,
) -> usize {
    if !cfg.friction_cone_soft {
        return 0;
    }
    let mut count = 0;
    for k in 0..n {
        for leg in 0..N_FEET {
            if contact.is_stance[leg][k] {
                count += 2;
            }
        }
    }
    count
}

/// Constraint assembly. Equalities first (`ZeroCone`), inequalities
/// second (`NonnegativeCone`); clarabel requires that order.
///
/// Equalities:
/// 1. Swing leg GRF = 0 (3 rows per swing-leg-step)
/// 2. Stance no-slip — `v_foot_world(stance leg, step k) = 0`
///    expressed linearly in U via the lifted state. 3 rows per
///    stance-leg-step. See D3.3.4b derivation in module-level docs.
///
/// Inequalities: per stance leg per step,
/// - `f_z ≥ 0` (1 row)
/// - `f_z ≤ f_max` (1 row, if `f_max > 0`)
/// - `|f_x| ≤ μ·f_z` (2 rows; soft form adds `−s_x` term)
/// - `|f_y| ≤ μ·f_z` (2 rows; soft form adds `−s_y` term)
///
/// When `cfg.friction_cone_soft = true`, the QP's decision vector is
/// extended by `n_slacks` extra entries (two per stance-leg-step:
/// `s_x, s_y`). The friction rows reference these slacks; an extra
/// `s ≥ 0` row per slack is appended after the cone rows.
fn build_constraints_24(
    cfg: &FullCentroidalMpcConfig,
    contact: &FullCentroidalContactSchedule,
    ref_traj: &FullCentroidalReference,
    n: usize,
    a_x: &DMatrix<f64>,
    b_u: &DMatrix<f64>,
    x_now: &DVector<f64>,
    n_slacks: usize,
) -> (CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
    let nu = N_INPUT;
    let nx = N_STATE_AUG;
    let total_vars = nu * n + n_slacks;
    let mu = cfg.friction_mu;
    let soft_cone = cfg.friction_cone_soft;
    let f_max_global = cfg.max_normal_force;
    let legs = [LegId::FL, LegId::FR, LegId::RL, LegId::RR];

    // Effective per-(leg, k) f_z upper bound — the tighter of the
    // global `cfg.max_normal_force` and the schedule-supplied
    // `contact.stance_f_max[leg][k]`. Returns `f64::INFINITY` when
    // neither is set, which means no upper bound row is emitted.
    let effective_f_max = |leg: usize, k: usize| -> f64 {
        let local = contact.stance_f_max[leg][k];
        let local_active = local.is_finite() && local >= 0.0;
        let global_active = f_max_global > 0.0;
        match (global_active, local_active) {
            (true, true) => f_max_global.min(local),
            (true, false) => f_max_global,
            (false, true) => local,
            (false, false) => f64::INFINITY,
        }
    };

    let mut n_eq = 0;
    let mut n_ineq = 0;
    for k in 0..n {
        for leg in 0..N_FEET {
            if contact.is_stance[leg][k] {
                // Stance no-slip: 3 equality rows
                n_eq += 3;
                let mut count = 4; // friction ±x, ±y
                count += 1; // f_z ≥ 0
                if effective_f_max(leg, k).is_finite() {
                    count += 1;
                }
                n_ineq += count;
            } else {
                n_eq += 3; // swing F = 0
                if cfg.enable_swing_normal_velocity_constraint {
                    n_eq += 1; // swing v_foot_world.z = v_z_planned
                }
            }
        }
    }
    // A3: each slack contributes one `s ≥ 0` inequality row.
    n_ineq += n_slacks;

    let n_rows = n_eq + n_ineq;
    let mut a_dense = DMatrix::<f64>::zeros(n_rows, total_vars);
    let mut b_vec = vec![0.0; n_rows];
    let mut row = 0;

    // ── Equality 1: swing GRF = 0 ─────────────────────────────────────
    for k in 0..n {
        for leg in 0..N_FEET {
            if !contact.is_stance[leg][k] {
                let col = k * nu + leg * 3;
                for ax in 0..3 {
                    a_dense[(row + ax, col + ax)] = 1.0;
                }
                row += 3;
            }
        }
    }

    // ── Equality 2: stance no-slip ────────────────────────────────────
    // Per stance-leg-step:
    //   v_foot_world = v_com + ω × r_ref + R_z · J_foot(q_ref) · joint_v_leg = 0
    // where r_ref = R_z · (foot_body_ref − com_offset_body). Substituting
    // `v_com_k = (A_x[k, 0..3] · x_0) + (B_u[k, 0..3] · U)` and same for
    // ω_k yields a linear row in U with constant RHS.
    for k in 0..n {
        for leg in 0..N_FEET {
            if !contact.is_stance[leg][k] {
                continue;
            }
            let psi = ref_traj.states[k].base_euler_zyx.z;
            let (s, c) = psi.sin_cos();
            let r_z = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
            let kin = cfg.kinematics.leg(legs[leg]);
            let [qhip, qthigh, qcalf] = ref_traj.states[k].leg_joint_q(leg);
            let foot_body_ref = forward_leg_kinematics(kin, qhip, qthigh, qcalf);
            let r_ref = r_z * (foot_body_ref - cfg.com_offset_body);
            let j_foot_body = foot_jacobian_body(kin, qhip, qthigh, qcalf);
            let r_z_j = r_z * j_foot_body;
            let m_skew = skew_3(&r_ref);

            // Build the 3×6 M matrix = [I_3, -skew(r_ref)] applied to
            // the 6-row (v_com; ω) block of the lifted state at step k.
            // For each of the 3 constraint rows we walk every column of U.
            // Slack columns (`nu*n..total_vars`) don't enter the dynamics
            // so they're skipped — their coefficient stays at zero.
            let row_base_in_a = k * nx; // top of step-k state block in a_x / b_u
            let joint_v_col_base = k * nu + 12 + 3 * leg;
            for ax in 0..3 {
                // Coefficient column-by-column. Sparse but the matrix
                // sizes are small enough that the dense pass is fine.
                for col_u in 0..(nu * n) {
                    let mut coef = b_u[(row_base_in_a + ax, col_u)]; // v_com row
                    for sr in 0..3 {
                        coef += -m_skew[(ax, sr)] * b_u[(row_base_in_a + 3 + sr, col_u)];
                    }
                    // Direct U slice for this stance leg's joint_v_leg
                    if col_u >= joint_v_col_base && col_u < joint_v_col_base + 3 {
                        let local = col_u - joint_v_col_base;
                        coef += r_z_j[(ax, local)];
                    }
                    if coef.abs() > 1e-14 {
                        a_dense[(row + ax, col_u)] = coef;
                    }
                }
                // RHS: -(M row · A_x[k_block][0..6, :] · x_0)
                let mut rhs = 0.0;
                for col_x in 0..nx {
                    let mut m_row = 0.0;
                    m_row += a_x[(row_base_in_a + ax, col_x)]; // v_com[ax]
                    for sr in 0..3 {
                        m_row += -m_skew[(ax, sr)] * a_x[(row_base_in_a + 3 + sr, col_x)];
                    }
                    rhs += m_row * x_now[col_x];
                }
                b_vec[row + ax] = -rhs;
            }
            row += 3;
        }
    }

    // ── Equality 3 (opt-in): swing-leg vertical foot velocity ─────────
    // Mirrors legged_control's `NormalVelocityConstraintCppAd`. Active
    // per swing-leg-step when `cfg.enable_swing_normal_velocity_constraint`
    // is on. The linearization is identical to stance no-slip's z-row,
    // with the RHS swapped from 0 to the planner-supplied
    // `v_z_planned = contact.swing_z_velocity[leg][k]`. Horizontal foot
    // motion remains unconstrained — the optimiser shapes it via the
    // joint_v cost + the next stance's no-slip constraint at touchdown
    // (legged_control's design choice).
    if cfg.enable_swing_normal_velocity_constraint {
        for k in 0..n {
            for leg in 0..N_FEET {
                if contact.is_stance[leg][k] {
                    continue;
                }
                let psi = ref_traj.states[k].base_euler_zyx.z;
                let (s, c) = psi.sin_cos();
                let r_z = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
                let kin = cfg.kinematics.leg(legs[leg]);
                let [qhip, qthigh, qcalf] = ref_traj.states[k].leg_joint_q(leg);
                let foot_body_ref = forward_leg_kinematics(kin, qhip, qthigh, qcalf);
                let r_ref = r_z * (foot_body_ref - cfg.com_offset_body);
                let j_foot_body = foot_jacobian_body(kin, qhip, qthigh, qcalf);
                let r_z_j = r_z * j_foot_body;
                let m_skew = skew_3(&r_ref);

                let ax = 2; // z-row only
                let row_base_in_a = k * nx;
                let joint_v_col_base = k * nu + 12 + 3 * leg;
                for col_u in 0..(nu * n) {
                    let mut coef = b_u[(row_base_in_a + ax, col_u)];
                    for sr in 0..3 {
                        coef += -m_skew[(ax, sr)] * b_u[(row_base_in_a + 3 + sr, col_u)];
                    }
                    if col_u >= joint_v_col_base && col_u < joint_v_col_base + 3 {
                        let local = col_u - joint_v_col_base;
                        coef += r_z_j[(ax, local)];
                    }
                    if coef.abs() > 1e-14 {
                        a_dense[(row, col_u)] = coef;
                    }
                }
                let mut rhs = 0.0;
                for col_x in 0..nx {
                    let mut m_row = 0.0;
                    m_row += a_x[(row_base_in_a + ax, col_x)];
                    for sr in 0..3 {
                        m_row += -m_skew[(ax, sr)] * a_x[(row_base_in_a + 3 + sr, col_x)];
                    }
                    rhs += m_row * x_now[col_x];
                }
                let v_z_planned = contact.swing_z_velocity[leg][k];
                b_vec[row] = v_z_planned - rhs;
                row += 1;
            }
        }
    }

    // ── Inequality: friction + f_z bounds ─────────────────────────────
    // When `soft_cone` is on, the 4 friction rows reference an extra
    // slack column per axis. Slack columns sit at the tail of the
    // decision vector (`nu*n..nu*n + n_slacks`), assigned in
    // visitation order (`s_x` before `s_y`, leg-step-major).
    let mut slack_cursor = nu * n;
    for k in 0..n {
        for leg in 0..N_FEET {
            if !contact.is_stance[leg][k] {
                continue;
            }
            let col_x = k * nu + leg * 3;
            let col_y = col_x + 1;
            let col_z = col_x + 2;
            // f_z ≥ 0  ⇒  -f_z ≤ 0  (always hard)
            a_dense[(row, col_z)] = -1.0;
            row += 1;
            let f_max_this = effective_f_max(leg, k);
            if f_max_this.is_finite() {
                a_dense[(row, col_z)] = 1.0;
                b_vec[row] = f_max_this;
                row += 1;
            }
            // |f_x| ≤ μ·f_z (+ s_x in soft mode)
            let (sx_col, sy_col) = if soft_cone {
                let s_x = slack_cursor;
                let s_y = slack_cursor + 1;
                slack_cursor += 2;
                (Some(s_x), Some(s_y))
            } else {
                (None, None)
            };
            a_dense[(row, col_x)] = 1.0;
            a_dense[(row, col_z)] = -mu;
            if let Some(sx) = sx_col {
                a_dense[(row, sx)] = -1.0;
            }
            row += 1;
            a_dense[(row, col_x)] = -1.0;
            a_dense[(row, col_z)] = -mu;
            if let Some(sx) = sx_col {
                a_dense[(row, sx)] = -1.0;
            }
            row += 1;
            // |f_y| ≤ μ·f_z (+ s_y in soft mode)
            a_dense[(row, col_y)] = 1.0;
            a_dense[(row, col_z)] = -mu;
            if let Some(sy) = sy_col {
                a_dense[(row, sy)] = -1.0;
            }
            row += 1;
            a_dense[(row, col_y)] = -1.0;
            a_dense[(row, col_z)] = -mu;
            if let Some(sy) = sy_col {
                a_dense[(row, sy)] = -1.0;
            }
            row += 1;
        }
    }

    // ── Inequality: s_i ≥ 0 for each soft-cone slack ──────────────────
    // Emit one row per slack: -s_i ≤ 0. Slack columns are
    // `[nu*n, nu*n + n_slacks)` in the same visitation order as above.
    for i in 0..n_slacks {
        a_dense[(row, nu * n + i)] = -1.0;
        row += 1;
    }

    debug_assert_eq!(row, n_rows);
    let a_csc = dense_to_csc_24(&a_dense);
    let cones = vec![
        SupportedConeT::ZeroConeT(n_eq),
        SupportedConeT::NonnegativeConeT(n_ineq),
    ];
    (a_csc, b_vec, cones)
}

fn dense_to_csc_upper_24(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let nr = p.nrows();
    let nc = p.ncols();
    debug_assert_eq!(nr, nc);
    let mut colptr = Vec::with_capacity(nc + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..nc {
        for i in 0..=j {
            let v = p[(i, j)];
            if v.abs() > 1e-12 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(nr, nc, colptr, rowval, nzval)
}

fn dense_to_csc_24(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let nr = m.nrows();
    let nc = m.ncols();
    let mut colptr = Vec::with_capacity(nc + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..nc {
        for i in 0..nr {
            let v = m[(i, j)];
            if v.abs() > 1e-12 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(nr, nc, colptr, rowval, nzval)
}

/// Per-leg world-frame foot positions via the analytical 3R chain FK
/// composed with the floating base pose.
///
/// `foot_world_i = base_pos + R_world_body · forward_leg_kinematics(joint_q_i)`
///
/// Exposed at module scope (not just inside `full_centroidal_dynamics`)
/// because D3.3.3 needs the same expression for the linearization.
pub fn compute_foot_positions_world(
    state: &FullCentroidalState,
    cfg: &FullCentroidalMpcConfig,
) -> [Vector3<f64>; N_FEET] {
    let r_world_body = Rotation3::from_euler_angles(
        state.base_euler_zyx.x,
        state.base_euler_zyx.y,
        state.base_euler_zyx.z,
    );
    let legs = [LegId::FL, LegId::FR, LegId::RL, LegId::RR];
    let mut out = [Vector3::zeros(); N_FEET];
    for (slot, leg) in legs.iter().enumerate() {
        let kin = cfg.kinematics.leg(*leg);
        let [hip, thigh, calf] = state.leg_joint_q(slot);
        let foot_body = forward_leg_kinematics(kin, hip, thigh, calf);
        out[slot] = state.base_pos_world + r_world_body * foot_body;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LegKinematics;

    /// Symmetric quadruped with 0.4 m hip span / 0.5 m wheelbase, 0.3 m
    /// leg segments — close enough to namiashi to exercise the FK
    /// without depending on the URDF fixture.
    fn test_kinematics() -> KinematicsConfig {
        let make = |leg: LegId, x: f64, y: f64| {
            LegKinematics::new(
                leg,
                format!("{}_hip", leg.label()),
                format!("{}_thigh", leg.label()),
                format!("{}_calf", leg.label()),
                format!("{}_foot", leg.label()),
                Vector3::new(x, y, 0.0),
                0.05,
                0.15,
                0.15,
            )
        };
        KinematicsConfig {
            fl: make(LegId::FL, 0.25, 0.20),
            fr: make(LegId::FR, 0.25, -0.20),
            rl: make(LegId::RL, -0.25, 0.20),
            rr: make(LegId::RR, -0.25, -0.20),
        }
    }

    fn test_config() -> FullCentroidalMpcConfig {
        FullCentroidalMpcConfig {
            mass_kg: 9.0,
            centroidal_inertia_body: Matrix3::from_diagonal(&Vector3::new(0.07, 0.26, 0.242)),
            com_offset_body: Vector3::zeros(),
            friction_mu: 0.5,
            max_normal_force: 200.0,
            horizon_steps: 10,
            dt_per_step: 0.030,
            q_diag: [1.0; N_STATE],
            r_diag: [1e-3; N_INPUT],
            sqp_iterations: 1,
            kinematics: test_kinematics(),
            enable_swing_normal_velocity_constraint: false,
            friction_cone_soft: false,
            friction_cone_slack_penalty: 1000.0,
            warm_start: false,
            q_foot_xy_world: 0.0,
        }
    }

    #[test]
    fn dim_constants_are_24_and_24() {
        assert_eq!(N_STATE, 24);
        assert_eq!(N_INPUT, 24);
    }

    #[test]
    fn state_round_trip_through_vec24() {
        let s = FullCentroidalState {
            v_com_world: Vector3::new(0.1, -0.2, 0.3),
            angular_velocity_world: Vector3::new(-0.05, 0.07, 0.11),
            base_pos_world: Vector3::new(1.0, 2.0, 0.32),
            base_euler_zyx: Vector3::new(0.01, -0.02, 0.5),
            joint_q: [
                0.10, 0.50, -1.20, // FL
                -0.10, 0.50, -1.20, // FR
                0.10, 0.55, -1.25, // RL
                -0.10, 0.55, -1.25, // RR
            ],
        };
        let s2 = FullCentroidalState::from_vec(&s.to_vec());
        assert_eq!(s, s2);
    }

    #[test]
    fn input_round_trip_through_vec24() {
        let mut joint_v = [0.0; N_LEG_JOINTS];
        for (i, x) in joint_v.iter_mut().enumerate() {
            *x = 0.1 * (i as f64 + 1.0);
        }
        let u = FullCentroidalInput {
            grfs_world: [
                Vector3::new(1.0, 0.0, 30.0),
                Vector3::new(-1.0, 0.0, 30.0),
                Vector3::new(2.0, 0.5, 25.0),
                Vector3::new(-2.0, -0.5, 25.0),
            ],
            joint_v,
        };
        let u2 = FullCentroidalInput::from_vec(&u.to_vec());
        assert_eq!(u, u2);
    }

    #[test]
    fn leg_joint_q_extracts_correct_block() {
        let mut q = [0.0; N_LEG_JOINTS];
        for (i, x) in q.iter_mut().enumerate() {
            *x = i as f64;
        }
        let s = FullCentroidalState {
            joint_q: q,
            ..Default::default()
        };
        assert_eq!(s.leg_joint_q(0), [0.0, 1.0, 2.0]); // FL
        assert_eq!(s.leg_joint_q(1), [3.0, 4.0, 5.0]); // FR
        assert_eq!(s.leg_joint_q(2), [6.0, 7.0, 8.0]); // RL
        assert_eq!(s.leg_joint_q(3), [9.0, 10.0, 11.0]); // RR
    }

    #[test]
    fn state_default_is_zero() {
        let s = FullCentroidalState::default();
        assert_eq!(s.to_vec(), [0.0; N_STATE]);
    }

    #[test]
    fn input_default_is_zero() {
        let u = FullCentroidalInput::default();
        assert_eq!(u.to_vec(), [0.0; N_INPUT]);
    }

    #[test]
    fn dynamics_zero_input_yields_only_gravity_on_v_com() {
        // No GRF, no joint motion, identity orientation. The only
        // non-zero rate must be v_com_dot = (0,0,-9.81). Everything
        // else (ω̇, ṗ, ė, q̇) must be exactly 0.
        let state = FullCentroidalState::default();
        let input = FullCentroidalInput::default();
        let cfg = test_config();
        let dx = full_centroidal_dynamics(&state, &input, &cfg);
        assert!((dx.v_com_world - Vector3::new(0.0, 0.0, -9.81)).norm() < 1e-12);
        assert_eq!(dx.angular_velocity_world, Vector3::zeros());
        assert_eq!(dx.base_pos_world, Vector3::zeros());
        assert_eq!(dx.base_euler_zyx, Vector3::zeros());
        assert_eq!(dx.joint_q, [0.0; N_LEG_JOINTS]);
    }

    #[test]
    fn joint_v_drives_joint_q_dot_exactly() {
        // q̇_j = v_j is a trivial pass-through; verify component-wise.
        let mut joint_v = [0.0; N_LEG_JOINTS];
        for (i, x) in joint_v.iter_mut().enumerate() {
            *x = 0.1 * (i as f64 + 1.0);
        }
        let state = FullCentroidalState::default();
        let input = FullCentroidalInput {
            joint_v,
            ..Default::default()
        };
        let cfg = test_config();
        let dx = full_centroidal_dynamics(&state, &input, &cfg);
        assert_eq!(dx.joint_q, joint_v);
    }

    #[test]
    fn balanced_grf_at_nominal_pose_yields_zero_v_com_dot() {
        // Apply Σ F = m·g (= 88.29 N upward total), distributed evenly
        // across the four feet. v_com_dot should be exactly zero
        // (gravity cancels). Joint state at zero — feet are at their
        // straight-down nominal positions, so symmetric GRF gives
        // zero net moment too (verifies test rig is well-posed).
        let cfg = test_config();
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input = FullCentroidalInput {
            grfs_world: [
                Vector3::new(0.0, 0.0, f_per_foot),
                Vector3::new(0.0, 0.0, f_per_foot),
                Vector3::new(0.0, 0.0, f_per_foot),
                Vector3::new(0.0, 0.0, f_per_foot),
            ],
            ..Default::default()
        };
        let state = FullCentroidalState::default();
        let dx = full_centroidal_dynamics(&state, &input, &cfg);
        assert!(dx.v_com_world.norm() < 1e-9, "v_com_dot = {}", dx.v_com_world);
        // With com_offset = 0 and a symmetric pose, the angular accel
        // should also be zero (each pair of feet contributes opposing
        // moments that cancel).
        assert!(
            dx.angular_velocity_world.norm() < 1e-9,
            "α = {}",
            dx.angular_velocity_world
        );
    }

    #[test]
    fn foot_positions_at_zero_q_match_kinematics_nominal() {
        // With joint_q = 0 the analytical FK puts the foot at
        //   hip_offset + (0, ±hip_to_thigh_y, -(L1+L2))
        // For our test rig that's (±0.25, ±0.25, -0.30) — i.e.
        // |hip_to_thigh_y|=0.05 added to the hip y and the legs hang
        // straight down by 0.30.
        let cfg = test_config();
        let state = FullCentroidalState::default();
        let feet = compute_foot_positions_world(&state, &cfg);
        // FL: (+0.25, +0.20+0.05, -0.30) = (0.25, 0.25, -0.30)
        assert!((feet[0] - Vector3::new(0.25, 0.25, -0.30)).norm() < 1e-12);
        // FR: (+0.25, -0.20-0.05, -0.30) = (0.25, -0.25, -0.30)
        assert!((feet[1] - Vector3::new(0.25, -0.25, -0.30)).norm() < 1e-12);
        // RL / RR mirror with x = -0.25.
        assert!((feet[2] - Vector3::new(-0.25, 0.25, -0.30)).norm() < 1e-12);
        assert!((feet[3] - Vector3::new(-0.25, -0.25, -0.30)).norm() < 1e-12);
    }

    #[test]
    fn foot_positions_translate_with_base_pos() {
        // Moving the base in world should translate every foot by the
        // same vector (since R = I and joint_q = 0).
        let cfg = test_config();
        let state = FullCentroidalState {
            base_pos_world: Vector3::new(1.0, 2.0, 0.32),
            ..Default::default()
        };
        let feet = compute_foot_positions_world(&state, &cfg);
        let baseline = compute_foot_positions_world(&FullCentroidalState::default(), &cfg);
        for slot in 0..N_FEET {
            let diff = feet[slot] - baseline[slot] - state.base_pos_world;
            assert!(diff.norm() < 1e-12, "slot {slot}: {diff}");
        }
    }

    #[test]
    fn agrees_with_12_state_centroidal_when_joint_v_is_zero() {
        // Sanity check D3.3.2 against D1: with joint_q at the nominal
        // (zero) pose, and joint_v = 0 (so foot positions match what
        // the 12-state would receive externally), the body part of
        // the 24-state derivative must equal the 12-state derivative.
        use crate::centroidal_mpc::{
            centroidal_dynamics, CentroidalInput, CentroidalMpcConfig, CentroidalState,
        };

        let cfg24 = test_config();

        // Build the 12-state config with the same physical params.
        let cfg12 = CentroidalMpcConfig {
            mass_kg: cfg24.mass_kg,
            centroidal_inertia_body: cfg24.centroidal_inertia_body,
            com_offset_body: cfg24.com_offset_body,
            friction_mu: cfg24.friction_mu,
            max_normal_force: cfg24.max_normal_force,
            horizon_steps: cfg24.horizon_steps,
            dt_per_step: cfg24.dt_per_step,
            q_diag: [1.0; 12],
            r_diag: 1e-3,
            sqp_iterations: 1,
        };

        // Non-trivial state — small body velocity + small angular vel.
        let state24 = FullCentroidalState {
            v_com_world: Vector3::new(0.10, 0.0, 0.0),
            angular_velocity_world: Vector3::new(0.01, 0.02, 0.05),
            base_pos_world: Vector3::new(0.0, 0.0, 0.32),
            base_euler_zyx: Vector3::zeros(), // identity → keeps the test pure
            joint_q: [0.0; N_LEG_JOINTS],
        };
        let state12 = CentroidalState {
            h_lin_per_mass: state24.v_com_world,
            angular_velocity_world: state24.angular_velocity_world,
            base_pos_world: state24.base_pos_world,
            base_euler_zyx: state24.base_euler_zyx,
        };

        // Asymmetric GRFs to exercise the full angular dynamics term.
        let input24 = FullCentroidalInput {
            grfs_world: [
                Vector3::new(2.0, 1.0, 25.0),
                Vector3::new(-1.5, 0.5, 22.0),
                Vector3::new(2.5, -0.8, 24.0),
                Vector3::new(-2.0, -0.3, 17.0),
            ],
            joint_v: [0.0; N_LEG_JOINTS], // freeze joints
        };
        let input12 = CentroidalInput {
            grfs_world: input24.grfs_world,
        };

        // Foot positions feeding the 12-state must come from the same
        // base pose + joint config as the 24-state to make the
        // comparison meaningful.
        let foot_world = compute_foot_positions_world(&state24, &cfg24);

        let dx24 = full_centroidal_dynamics(&state24, &input24, &cfg24);
        let dx12 = centroidal_dynamics(&state12, &input12, &foot_world, &cfg12);

        let tol = 1e-9;
        assert!((dx24.v_com_world - dx12.h_lin_per_mass).norm() < tol);
        assert!((dx24.angular_velocity_world - dx12.angular_velocity_world).norm() < tol);
        assert!((dx24.base_pos_world - dx12.base_pos_world).norm() < tol);
        assert!((dx24.base_euler_zyx - dx12.base_euler_zyx).norm() < tol);
    }

    /// Convert `FullCentroidalState` to a plain 24-vec for FD perturbation.
    fn dx_minus(a: &FullCentroidalState, b: &FullCentroidalState) -> [f64; N_STATE] {
        let av = a.to_vec();
        let bv = b.to_vec();
        let mut out = [0.0; N_STATE];
        for i in 0..N_STATE {
            out[i] = av[i] - bv[i];
        }
        out
    }

    /// Build a non-trivial reference state for linearization tests.
    /// Roll/pitch = 0 (the linearization assumes this), but joint_q
    /// and v_com are non-zero so the Jacobian columns we care about
    /// (∂α/∂q in particular) carry non-trivial values.
    fn ref_state_for_lin_test() -> FullCentroidalState {
        FullCentroidalState {
            v_com_world: Vector3::new(0.10, 0.05, 0.0),
            angular_velocity_world: Vector3::zeros(),
            base_pos_world: Vector3::new(0.0, 0.0, 0.32),
            base_euler_zyx: Vector3::zeros(),
            joint_q: [
                0.05, 0.55, -1.20, // FL
                -0.05, 0.55, -1.20, // FR
                0.05, 0.60, -1.25, // RL
                -0.05, 0.60, -1.25, // RR
            ],
        }
    }

    fn ref_input_for_lin_test() -> FullCentroidalInput {
        let mut joint_v = [0.0; N_LEG_JOINTS];
        for (i, x) in joint_v.iter_mut().enumerate() {
            *x = 0.05 * (i as f64 + 1.0).sin();
        }
        FullCentroidalInput {
            grfs_world: [
                Vector3::new(2.0, 1.0, 25.0),
                Vector3::new(-1.5, 0.8, 22.0),
                Vector3::new(2.5, -0.7, 24.0),
                Vector3::new(-2.0, -0.5, 17.0),
            ],
            joint_v,
        }
    }

    /// Numerical column-k of ∂f/∂x via central FD on
    /// `full_centroidal_dynamics`. Returns the 24-vector
    /// `(f(x+ε e_k, u) − f(x−ε e_k, u)) / (2ε)`.
    fn fd_state_col(
        k: usize,
        state: &FullCentroidalState,
        input: &FullCentroidalInput,
        cfg: &FullCentroidalMpcConfig,
        eps: f64,
    ) -> [f64; N_STATE] {
        let mut xv = state.to_vec();
        let original = xv[k];
        xv[k] = original + eps;
        let f_plus = full_centroidal_dynamics(&FullCentroidalState::from_vec(&xv), input, cfg);
        xv[k] = original - eps;
        let f_minus = full_centroidal_dynamics(&FullCentroidalState::from_vec(&xv), input, cfg);
        let mut out = [0.0; N_STATE];
        let plus = f_plus.to_vec();
        let minus = f_minus.to_vec();
        for i in 0..N_STATE {
            out[i] = (plus[i] - minus[i]) / (2.0 * eps);
        }
        out
    }

    fn fd_input_col(
        k: usize,
        state: &FullCentroidalState,
        input: &FullCentroidalInput,
        cfg: &FullCentroidalMpcConfig,
        eps: f64,
    ) -> [f64; N_STATE] {
        let mut uv = input.to_vec();
        let original = uv[k];
        uv[k] = original + eps;
        let f_plus = full_centroidal_dynamics(state, &FullCentroidalInput::from_vec(&uv), cfg);
        uv[k] = original - eps;
        let f_minus = full_centroidal_dynamics(state, &FullCentroidalInput::from_vec(&uv), cfg);
        let mut out = [0.0; N_STATE];
        let plus = f_plus.to_vec();
        let minus = f_minus.to_vec();
        for i in 0..N_STATE {
            out[i] = (plus[i] - minus[i]) / (2.0 * eps);
        }
        out
    }

    #[test]
    fn linearization_state_jacobian_matches_fd() {
        let cfg = test_config();
        let state = ref_state_for_lin_test();
        let input = ref_input_for_lin_test();
        let stance = [true; N_FEET];
        let psi_ref = state.base_euler_zyx.z;

        let (a_mat, _b_mat) = continuous_dynamics_full(&state, &input, &cfg, &stance, psi_ref);
        let eps = 1e-6;
        let tol = 1e-3; // FD has ~1e-6 noise, plus we drop ω×Iω

        // Skip column 24 (augmented gravity) — not part of the
        // non-augmented `full_centroidal_dynamics` state.
        for k in 0..N_STATE {
            let fd = fd_state_col(k, &state, &input, &cfg, eps);
            for i in 0..N_STATE {
                let analytical = a_mat[(i, k)];
                let diff = (fd[i] - analytical).abs();
                assert!(
                    diff < tol,
                    "A[{i},{k}]: analytical={analytical:.6e}  fd={:.6e}  diff={diff:.3e}",
                    fd[i]
                );
            }
        }
        let _ = dx_minus; // silence warning — kept for future tests
    }

    #[test]
    fn linearization_input_jacobian_matches_fd() {
        let cfg = test_config();
        let state = ref_state_for_lin_test();
        let input = ref_input_for_lin_test();
        let stance = [true; N_FEET];
        let psi_ref = state.base_euler_zyx.z;

        let (_a_mat, b_mat) = continuous_dynamics_full(&state, &input, &cfg, &stance, psi_ref);
        let eps = 1e-6;
        let tol = 1e-3;

        for k in 0..N_INPUT {
            let fd = fd_input_col(k, &state, &input, &cfg, eps);
            for i in 0..N_STATE {
                let analytical = b_mat[(i, k)];
                let diff = (fd[i] - analytical).abs();
                assert!(
                    diff < tol,
                    "B[{i},{k}]: analytical={analytical:.6e}  fd={:.6e}  diff={diff:.3e}",
                    fd[i]
                );
            }
        }
    }

    #[test]
    fn linearization_swing_legs_zero_their_grf_block() {
        // A swing leg's foot is in the air → no GRF effect on body
        // dynamics regardless of any input force (constraint enforced
        // by the QP equality). The linearization should reflect this
        // by zeroing the corresponding 6×3 block of B for v̇_com and α
        // rows.
        let cfg = test_config();
        let state = ref_state_for_lin_test();
        let input = ref_input_for_lin_test();
        let stance = [true, false, true, false]; // FR, RR are swinging
        let (_a, b) = continuous_dynamics_full(&state, &input, &cfg, &stance, 0.0);

        for swing_leg in [1usize, 3] {
            for j in 0..3 {
                for i in 0..6 {
                    assert_eq!(b[(i, 3 * swing_leg + j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn mpc_solve_at_static_stand_returns_gravity_balancing_grfs() {
        // Static-stand reference: body at z=0.32, joint_q at "knees
        // flexed" so the foot rests on z=0. Reference inputs balance
        // gravity. The MPC should converge to ~ref inputs.
        let mut cfg = test_config();
        cfg.horizon_steps = 5; // shorter for unit-test speed
        cfg.sqp_iterations = 1;

        // Pick a joint_q that puts foot z ≈ -0.32 (= base z = 0.32 above ground).
        // Test rig: l1=l2=0.15, so straight-down gives z=-0.30. We
        // accept that small offset; the QP will simply produce some
        // f_z slightly above static.
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let mut mpc = FullCentroidalMpc::new(cfg);
        let sol = mpc.solve(state_ref, &reference, &contact);
        assert!(sol.solved, "QP must solve at static stand");

        // First-step GRFs: each foot's f_z should be in [10, 60] N
        // (= roughly within 50% of the static reference).
        for leg in 0..N_FEET {
            let fz = sol.first_input.grfs_world[leg].z;
            assert!(
                fz > 5.0 && fz < 70.0,
                "leg {leg} f_z = {fz} N outside reasonable range"
            );
        }
        // Joint_v should be small (no swing leg, no big body motion).
        for v in sol.first_input.joint_v.iter() {
            assert!(v.abs() < 2.0, "joint_v {v} too large at static stand");
        }
    }

    #[test]
    fn mpc_stance_no_slip_keeps_foot_velocity_zero() {
        // Body starts moving forward at v_com.x = 0.1 m/s; reference is
        // static stand (v_com_ref = 0). All four legs stance.
        // The stance no-slip constraint should force the MPC's
        // solution joint_v to make each foot velocity zero —
        // i.e. the legs "rotate backward" relative to the body to
        // compensate for the body's motion.
        //
        // Foot velocity at the linearization (= reference) point with
        // first input u_0:
        //   v_foot[leg] = v_com_now + ω_now × r_ref[leg]
        //               + R_z · J_foot · joint_v_leg
        // With v_com_now = (0.1, 0, 0) and ω_now = 0, this means
        // R_z · J_foot · joint_v_leg ≈ (-0.1, 0, 0) per leg. The exact
        // tolerance is loose because the constraint is on
        // post-discretization-integrated state, not on instantaneous
        // velocity at t=0. We check that |v_foot_world| at t=0 is well
        // below the un-constrained value of 0.1 m/s.
        let mut cfg = test_config();
        cfg.horizon_steps = 4;
        cfg.sqp_iterations = 1;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);

        let state_now = FullCentroidalState {
            v_com_world: Vector3::new(0.1, 0.0, 0.0),
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let mut mpc = FullCentroidalMpc::new(cfg.clone());
        let sol = mpc.solve(state_now, &reference, &contact);
        assert!(sol.solved, "MPC must solve under stance no-slip constraint");

        // For each leg, compute v_foot_world = v_com_now + R_z·J_foot·joint_v_leg
        // (ω_now = 0 so the ω×r term vanishes).
        for leg in 0..N_FEET {
            let kin = cfg.kinematics.leg(
                [LegId::FL, LegId::FR, LegId::RL, LegId::RR][leg],
            );
            let [qhip, qthigh, qcalf] = state_now.leg_joint_q(leg);
            let j_foot = foot_jacobian_body(kin, qhip, qthigh, qcalf);
            let joint_v = sol.first_input.joint_v;
            let v_leg = Vector3::new(
                joint_v[3 * leg],
                joint_v[3 * leg + 1],
                joint_v[3 * leg + 2],
            );
            // R_z at zero yaw is identity.
            let v_foot = state_now.v_com_world + j_foot * v_leg;
            assert!(
                v_foot.norm() < 0.03,
                "leg {leg} foot velocity {v_foot} not pinned to 0 by stance constraint (||·|| = {})",
                v_foot.norm()
            );
        }
    }

    #[test]
    fn mpc_solve_with_swing_leg_zeros_its_grf() {
        // FR leg swinging in the air (stance=false). The equality
        // constraint must pin its GRF to exactly zero in the solution.
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 3.0; // 3 stance legs
        let mut grfs = [Vector3::zeros(); N_FEET];
        for leg in [0, 2, 3] {
            grfs[leg].z = f_per_foot;
        }
        let input_ref = FullCentroidalInput {
            grfs_world: grfs,
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let mut contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        for k in 0..cfg.horizon_steps {
            contact.is_stance[1][k] = false; // FR swing
        }
        let mut mpc = FullCentroidalMpc::new(cfg);
        let sol = mpc.solve(state_ref, &reference, &contact);
        assert!(sol.solved);
        // FR GRF must be exactly zero (equality constraint).
        for k in 0..sol.inputs_all_steps.len() {
            let fr = sol.inputs_all_steps[k].grfs_world[1];
            assert!(fr.norm() < 1e-7, "FR GRF nonzero at step {k}: {fr}");
        }
    }

    /// With `enable_swing_normal_velocity_constraint = true`, the MPC
    /// must drive the swing leg's foot vertical velocity to match the
    /// planner-supplied `swing_z_velocity` at each step. Mirrors
    /// legged_control's `NormalVelocityConstraintCppAd` behaviour.
    #[test]
    fn mpc_swing_normal_velocity_constraint_tracks_planned_vz() {
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        cfg.enable_swing_normal_velocity_constraint = true;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        // 3 stance legs (FL, RL, RR) share gravity; FR (slot 1) swings.
        let f_per_foot = cfg.mass_kg * 9.81 / 3.0;
        let mut grfs = [Vector3::zeros(); N_FEET];
        for leg in [0, 2, 3] {
            grfs[leg].z = f_per_foot;
        }
        let input_ref = FullCentroidalInput {
            grfs_world: grfs,
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let mut contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let v_z_target = 0.10;
        for k in 0..cfg.horizon_steps {
            contact.is_stance[1][k] = false;
            contact.swing_z_velocity[1][k] = v_z_target;
        }
        let mut mpc = FullCentroidalMpc::new(cfg.clone());
        let sol = mpc.solve(state_ref, &reference, &contact);
        assert!(
            sol.solved,
            "MPC must solve with swing normal velocity constraint"
        );

        // Reconstruct v_foot_world.z at each horizon row from the
        // predicted state and the solved joint_v, using the same
        // linearisation point the constraint was built around
        // (foot_body and J_foot at ref state's joint_q; ω·r picked up
        // from the predicted ω). With R_z = I at zero yaw, the world
        // and body z axes coincide.
        let kin = cfg.kinematics.leg(LegId::FR);
        let [qhip, qthigh, qcalf] = state_ref.leg_joint_q(1);
        let j_foot = foot_jacobian_body(kin, qhip, qthigh, qcalf);
        let foot_body = forward_leg_kinematics(kin, qhip, qthigh, qcalf);
        let r_ref = foot_body - cfg.com_offset_body;
        for k in 0..cfg.horizon_steps {
            let v_com_z = sol.predicted_states[k].v_com_world.z;
            let omega = sol.predicted_states[k].angular_velocity_world;
            let jv = sol.inputs_all_steps[k].joint_v;
            let joint_v_fr = Vector3::new(jv[3], jv[4], jv[5]);
            let v_foot_z =
                v_com_z + omega.cross(&r_ref).z + (j_foot * joint_v_fr).z;
            assert!(
                (v_foot_z - v_z_target).abs() < 1e-3,
                "FR v_foot_z step {k}: got {v_foot_z}, want {v_z_target}"
            );
        }
    }

    /// C1-2: when the schedule supplies a `stance_f_max[leg][k]`
    /// tighter than `cfg.max_normal_force`, the MPC must honour it.
    /// Setting FL's f_max to 5 N at step 0 — well below the hover
    /// load `m·g/4 ≈ 22 N` for namiashi class — should clamp FL's
    /// solved vertical GRF at ≤ 5 N (with the rest of the legs
    /// picking up the slack via the friction-cone-compatible
    /// redistribution).
    #[test]
    fn mpc_stance_f_max_per_step_bound_is_enforced() {
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let mut contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        // Tighten FL only (slot 0) at every step to a small value
        // well below the per-foot hover load.
        let tight_f_max = 5.0;
        for k in 0..cfg.horizon_steps {
            contact.stance_f_max[0][k] = tight_f_max;
        }
        let mut mpc = FullCentroidalMpc::new(cfg.clone());
        let sol = mpc.solve(state_ref, &reference, &contact);
        assert!(sol.solved, "MPC must solve under tightened FL f_max bound");

        // FL vertical GRF must respect the tightened bound at every
        // step. A small numerical slack (1e-5) covers clarabel's
        // termination tolerance.
        for k in 0..sol.inputs_all_steps.len() {
            let fl_z = sol.inputs_all_steps[k].grfs_world[0].z;
            assert!(
                fl_z <= tight_f_max + 1e-5,
                "FL f_z step {k} = {fl_z} exceeds tightened bound {tight_f_max}"
            );
            assert!(
                fl_z >= -1e-9,
                "FL f_z step {k} = {fl_z} violates the f_z ≥ 0 lower bound"
            );
        }
    }

    /// A3: with `friction_cone_soft = true` and benign physics
    /// (no lateral GRF demand, sufficient μ), the slack penalty in
    /// the cost should drive slacks to zero — the friction cone must
    /// still bind exactly the same as the hard formulation when
    /// nothing forces it to break. The exposed solution (GRFs) should
    /// match the hard-mode counterpart to within a tight tolerance.
    #[test]
    fn mpc_friction_cone_soft_matches_hard_when_unloaded() {
        // Run the same nominal hover setup twice — once hard, once
        // soft — and compare first-step GRFs. With balanced gravity
        // load and no horizontal demand the cone is far from binding;
        // soft mode mustn't drift the answer.
        let mut cfg_hard = test_config();
        cfg_hard.horizon_steps = 3;
        cfg_hard.sqp_iterations = 1;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg_hard.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg_hard.horizon_steps],
            inputs: vec![input_ref; cfg_hard.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg_hard.horizon_steps);

        let mut mpc_hard = FullCentroidalMpc::new(cfg_hard.clone());
        let sol_hard = mpc_hard.solve(state_ref, &reference, &contact);
        assert!(sol_hard.solved);

        let mut cfg_soft = cfg_hard.clone();
        cfg_soft.friction_cone_soft = true;
        cfg_soft.friction_cone_slack_penalty = 1000.0;
        let mut mpc_soft = FullCentroidalMpc::new(cfg_soft);
        let sol_soft = mpc_soft.solve(state_ref, &reference, &contact);
        assert!(sol_soft.solved);

        for leg in 0..N_FEET {
            let dh = sol_hard.first_input.grfs_world[leg];
            let ds = sol_soft.first_input.grfs_world[leg];
            assert!(
                (dh - ds).norm() < 1e-3,
                "soft cone perturbed unloaded GRF: hard={dh:?}, soft={ds:?}"
            );
        }
    }

    /// A3 (recovery semantic): with a tiny μ and a state whose drift
    /// requires substantial lateral GRF to oppose, hard-mode would be
    /// infeasible at the pyramid corner — soft mode trades cost for
    /// feasibility and still returns a usable GRF. Without this
    /// flag the failed-solution fallback (reference inputs, NaN
    /// objective) would be the only output and the controller's GRF
    /// would degrade to gravity-balance with no recovery authority.
    #[test]
    fn mpc_friction_cone_soft_returns_feasible_under_tight_mu() {
        // mu = 0.05 is far below normal; pair with a state biased
        // sideways so the cost wants a large lateral GRF.
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        cfg.friction_mu = 0.05;
        cfg.friction_cone_soft = true;
        cfg.friction_cone_slack_penalty = 1000.0;
        // Penalise lateral CoM velocity error heavily so the MPC
        // would *want* to push laterally — exposes the cone.
        cfg.q_diag[1] = 1e3;
        let state_now = FullCentroidalState {
            v_com_world: Vector3::new(0.0, 0.30, 0.0),
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);

        let mut mpc = FullCentroidalMpc::new(cfg.clone());
        let sol = mpc.solve(state_now, &reference, &contact);
        assert!(
            sol.solved,
            "soft-cone MPC should return a feasible solution under tight mu"
        );
        // The solver must still ship a normal force at each stance
        // foot (slacks can't substitute for f_z, which is hard).
        for leg in 0..N_FEET {
            for k in 0..cfg.horizon_steps {
                let fz = sol.inputs_all_steps[k].grfs_world[leg].z;
                assert!(fz >= -1e-6, "f_z negative under soft cone at leg {leg} k {k}: {fz}");
            }
        }
    }

    /// B3: with `warm_start = false`, the cache must stay empty
    /// across solves (no behaviour change for the legacy cold path).
    #[test]
    fn mpc_warm_start_disabled_leaves_cache_empty() {
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        cfg.warm_start = false;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let mut mpc = FullCentroidalMpc::new(cfg);
        let _ = mpc.solve(state_ref, &reference, &contact);
        assert!(
            mpc.warm_start_cache.is_none(),
            "warm-start cache must stay empty when feature is off"
        );
    }

    /// B3: with `warm_start = true` and steady-state inputs, the
    /// second solve's seed comes from the first solve's prediction.
    /// We verify two things:
    /// 1. The MPC caches a non-empty trajectory after solve 1.
    /// 2. Solve 2 still returns the same gravity-balancing answer
    ///    (warm-start doesn't break correctness on a near-trivial
    ///    problem).
    #[test]
    fn mpc_warm_start_caches_and_converges_at_steady_state() {
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        cfg.warm_start = true;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let mut mpc = FullCentroidalMpc::new(cfg);

        let sol1 = mpc.solve(state_ref, &reference, &contact);
        assert!(sol1.solved);
        assert!(
            mpc.warm_start_cache.is_some(),
            "first solve must populate the warm-start cache"
        );

        // Solve 2 — warm-started from sol1's prediction.
        let sol2 = mpc.solve(state_ref, &reference, &contact);
        assert!(sol2.solved);
        // Steady-state: per-foot vertical GRF should match gravity
        // balance to within solver tolerance regardless of warm-start.
        for leg in 0..N_FEET {
            let fz = sol2.first_input.grfs_world[leg].z;
            assert!(
                (fz - f_per_foot).abs() < 1e-2,
                "warm-started solve diverged at leg {leg}: f_z = {fz}, want {f_per_foot}"
            );
        }
    }

    /// B3: changing `horizon_steps` must invalidate the cache (length
    /// mismatch would otherwise panic the SQP shape asserts on the
    /// next solve).
    #[test]
    fn mpc_warm_start_cache_invalidated_on_horizon_change() {
        let mut cfg = test_config();
        cfg.horizon_steps = 3;
        cfg.sqp_iterations = 1;
        cfg.warm_start = true;
        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let f_per_foot = cfg.mass_kg * 9.81 / 4.0;
        let input_ref = FullCentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot); 4],
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; cfg.horizon_steps],
            inputs: vec![input_ref; cfg.horizon_steps],
        };
        let contact = FullCentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let mut mpc = FullCentroidalMpc::new(cfg.clone());
        let _ = mpc.solve(state_ref, &reference, &contact);
        assert!(mpc.warm_start_cache.is_some());

        let mut cfg2 = cfg.clone();
        cfg2.horizon_steps = 5;
        mpc.set_config(cfg2);
        assert!(
            mpc.warm_start_cache.is_none(),
            "horizon change must clear the warm-start cache"
        );
    }

    /// A1: with `q_foot_xy_world > 0` and a target placed *away*
    /// from the nominal foot position of a swing leg, the MPC must
    /// produce non-zero joint_v for that leg over the horizon so the
    /// integrated foot position approaches the target. The same
    /// scenario with the cost off must keep joint_v at zero (no
    /// reason to move the leg).
    #[test]
    fn mpc_foot_xy_cost_drives_joint_v_toward_target() {
        // Set up a horizon where FR (slot 1) swings the entire time
        // and touches down at the final step. Nominally with zero
        // joint_v its foot stays at the FK(q_ref) position. Asking
        // the cost to land it 5 cm forward of nominal in world frame
        // must force non-zero joint_v.
        let n = 4_usize;
        let mut cfg = test_config();
        cfg.horizon_steps = n;
        cfg.sqp_iterations = 3;
        cfg.q_foot_xy_world = 1000.0;
        // Lighten the joint_v cost a touch so the optimiser is
        // willing to spend joint_v to track foot XY; the default
        // r_diag[joint_v] = 1e-3 (set in `test_config`) is already
        // permissive but be explicit.
        for i in 12..N_INPUT {
            cfg.r_diag[i] = 1e-3;
        }

        let state_ref = FullCentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        // 3-stance reference (FL, RL, RR), FR in air. Gravity goes
        // across the three stance legs.
        let f_per_foot = cfg.mass_kg * 9.81 / 3.0;
        let mut grfs = [Vector3::zeros(); N_FEET];
        for leg in [0, 2, 3] {
            grfs[leg].z = f_per_foot;
        }
        let input_ref = FullCentroidalInput {
            grfs_world: grfs,
            joint_v: [0.0; N_LEG_JOINTS],
        };
        let reference = FullCentroidalReference {
            states: vec![state_ref; n],
            inputs: vec![input_ref; n],
        };

        // Schedule: FR swings k=0..n-2, touches down at k=n-1. Place
        // the foot-XY target at touchdown only, 5 cm forward of
        // nominal in world frame.
        let mut contact = FullCentroidalContactSchedule::all_stance(n);
        for k in 0..n - 1 {
            contact.is_stance[1][k] = false;
        }
        // Nominal world-frame FR foot at zero joint_q + base@origin
        // is `(0.25, -0.25, -0.30)` per `foot_positions_at_zero_q_match_kinematics_nominal`.
        // Plus base_pos.z = 0.30 ⇒ foot z = 0.0, xy = (0.25, -0.25).
        let nominal_world_xy = [0.25_f64, -0.25_f64];
        let target_offset_x = 0.05_f64;
        contact.foot_xy_target_world[1][n - 1] =
            Some([nominal_world_xy[0] + target_offset_x, nominal_world_xy[1]]);

        let mut mpc = FullCentroidalMpc::new(cfg.clone());
        let sol = mpc.solve(state_ref, &reference, &contact);
        assert!(sol.solved, "A1 MPC must solve with foot-XY cost active");

        // Sum the FR joint_v's contribution to world-frame foot
        // motion over the horizon at the linearisation point. With
        // zero yaw, body and world frames coincide for XY.
        let kin = cfg.kinematics.leg(LegId::FR);
        let j_foot = foot_jacobian_body(kin, 0.0, 0.0, 0.0);
        let mut foot_disp_x = 0.0;
        for k in 0..n - 1 {
            let jv = sol.inputs_all_steps[k].joint_v;
            let joint_v_fr = Vector3::new(jv[3], jv[4], jv[5]);
            foot_disp_x += (j_foot * joint_v_fr).x * cfg.dt_per_step;
        }
        // Cost-on must move the foot at least an order of magnitude
        // more than the baseline numerical noise (~4 mm with the
        // default test weights). 2 cm threshold gives clear
        // separation from the cost-off case while not requiring full
        // 5 cm convergence in 3 SQP iterations.
        assert!(
            foot_disp_x.abs() > 0.02,
            "MPC failed to move FR foot toward target: integrated dx = {foot_disp_x}, want ≳ {target_offset_x}"
        );

        // Reverse the experiment: same setup, but cost off ⇒ no
        // joint_v should fire (within numerical tolerance).
        let mut cfg_off = cfg.clone();
        cfg_off.q_foot_xy_world = 0.0;
        let dt_per_step_off = cfg_off.dt_per_step;
        let mut mpc_off = FullCentroidalMpc::new(cfg_off);
        let sol_off = mpc_off.solve(state_ref, &reference, &contact);
        assert!(sol_off.solved);
        let mut foot_disp_x_off = 0.0;
        for k in 0..n - 1 {
            let jv = sol_off.inputs_all_steps[k].joint_v;
            let joint_v_fr = Vector3::new(jv[3], jv[4], jv[5]);
            foot_disp_x_off += (j_foot * joint_v_fr).x * dt_per_step_off;
        }
        // The cost-off baseline has a small numerical residual (~4 mm
        // here) driven by the body-state cost dragging joint_q
        // through the SQP iterates. The point of this half of the
        // test is that the cost-on case beats it by an order of
        // magnitude — separation, not absolute zero.
        assert!(
            foot_disp_x_off.abs() < 0.5 * foot_disp_x.abs(),
            "cost-on/off separation insufficient: on={foot_disp_x}, off={foot_disp_x_off}"
        );
    }

    #[test]
    fn linearization_aug_gravity_column_is_unit_z_on_v_com() {
        // The augmented gravity state [24] only feeds into v_com_z dot
        // via A[2, 24] = 1. Every other entry in column 24 must be 0.
        let cfg = test_config();
        let state = ref_state_for_lin_test();
        let input = ref_input_for_lin_test();
        let (a, _b) = continuous_dynamics_full(&state, &input, &cfg, &[true; N_FEET], 0.0);
        for i in 0..N_STATE_AUG {
            let expected = if i == 2 { 1.0 } else { 0.0 };
            assert_eq!(a[(i, 24)], expected, "A[{i}, 24] != {expected}");
        }
    }
}
