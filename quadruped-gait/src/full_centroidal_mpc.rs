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

use nalgebra::{Matrix3, Rotation3, Vector3};

use crate::config::{KinematicsConfig, LegId};
use crate::ik::forward_leg_kinematics;

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
}
