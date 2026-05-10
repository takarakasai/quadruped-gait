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

use nalgebra::Vector3;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
