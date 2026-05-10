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
    /// Friction coefficient for the pyramid constraint.
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
#[derive(Clone, Debug)]
pub struct FullCentroidalContactSchedule {
    pub is_stance: [Vec<bool>; N_FEET],
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
}

impl FullCentroidalMpc {
    pub fn new(cfg: FullCentroidalMpcConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &FullCentroidalMpcConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: FullCentroidalMpcConfig) {
        self.cfg = cfg;
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
        }

        let n_iter = self.cfg.sqp_iterations.max(1);

        // SQP loop. Re-linearisation point updates from previous iter's
        // predicted (state, input) trajectory. Iter 0 starts at the
        // caller-supplied reference.
        let mut ref_traj = reference.clone();
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
        last_solution.expect("at least one SQP iteration ran")
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
        let p_dense = 2.0 * (b_u.transpose() * &q_block * &b_u + &r_block);
        let q_vec = 2.0 * (b_u.transpose() * &q_block * &drift - &r_block * &u_ref);

        // ── Constraints (D3.3.4a — swing GRF=0 + friction only) ──────
        let (a_csc, b_vec, cones) = build_constraints_24(&self.cfg, contact, n);

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
        let u_opt: Vec<f64> = solver.solution.x.clone();
        let objective = solver.solution.obj_val;

        // Decode U → per-step FullCentroidalInput.
        let mut inputs_all_steps = Vec::with_capacity(n);
        for k in 0..n {
            let base = k * nu;
            let mut slice = [0.0; N_INPUT];
            slice.copy_from_slice(&u_opt[base..base + nu]);
            inputs_all_steps.push(FullCentroidalInput::from_vec(&slice));
        }

        // Decode predicted states from X = A_x x_0 + B_u U.
        let u_dvec = DVector::from_vec(u_opt);
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

/// Constraint assembly (D3.3.4a): swing leg GRF = 0 (equality) +
/// friction pyramid + f_z bounds (inequality). Stance no-slip is added
/// in D3.3.4b.
fn build_constraints_24(
    cfg: &FullCentroidalMpcConfig,
    contact: &FullCentroidalContactSchedule,
    n: usize,
) -> (CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
    let nu = N_INPUT;
    let total_vars = nu * n;
    let mu = cfg.friction_mu;
    let f_max = cfg.max_normal_force;

    let mut n_eq = 0;
    let mut n_ineq = 0;
    for k in 0..n {
        for leg in 0..N_FEET {
            if contact.is_stance[leg][k] {
                let mut count = 4; // friction ±x, ±y
                count += 1; // f_z ≥ 0
                if f_max > 0.0 {
                    count += 1;
                }
                n_ineq += count;
            } else {
                n_eq += 3; // F = 0
            }
        }
    }

    let n_rows = n_eq + n_ineq;
    let mut a_dense = DMatrix::<f64>::zeros(n_rows, total_vars);
    let mut b_vec = vec![0.0; n_rows];
    let mut row = 0;

    // Equality: swing GRF = 0
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
    // Inequality: friction + f_z bounds
    for k in 0..n {
        for leg in 0..N_FEET {
            if !contact.is_stance[leg][k] {
                continue;
            }
            let col_x = k * nu + leg * 3;
            let col_y = col_x + 1;
            let col_z = col_x + 2;
            // f_z ≥ 0  ⇒  -f_z ≤ 0
            a_dense[(row, col_z)] = -1.0;
            row += 1;
            if f_max > 0.0 {
                a_dense[(row, col_z)] = 1.0;
                b_vec[row] = f_max;
                row += 1;
            }
            // |f_x| ≤ μ·f_z
            a_dense[(row, col_x)] = 1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
            a_dense[(row, col_x)] = -1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
            // |f_y| ≤ μ·f_z
            a_dense[(row, col_y)] = 1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
            a_dense[(row, col_y)] = -1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
        }
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
