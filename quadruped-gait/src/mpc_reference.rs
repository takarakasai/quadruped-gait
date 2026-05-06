//! Joint-level reference generator — articara's drop-in replacement
//! for `legged_control`'s OCS2 NMPC `optimized_state` → `pos_des`/
//! `vel_des` mapping.
//!
//! In `legged_control` the OCS2 NMPC predicts a centroidal state that
//! includes joint angles, so `pos_des(j) = getJointAngles(state)` is
//! a direct read-out of the optimised next-tick joint configuration.
//! Both Position-PD and the WBC's `formulateSwingLegTask` consume the
//! same `pos_des` / `vel_des`, which is why the two paths cooperate
//! instead of fighting.
//!
//! articara's SRBD MPC doesn't model joint state (its 13-dim state is
//! body-only: orientation, position, ω, v, g). To recover the same
//! "single source of truth" for joint references, we build them via a
//! cascade that's mathematically equivalent on the current tick:
//!
//! ```text
//! SRBD predicted body pose  +  footstep planner output   +   IK
//!         (next-tick)            (per-leg foot world)         (per-leg)
//!                       ↓
//!              JointReference { q*, q̇* }
//!                       ↓
//!         Position-PD reference   +   WBC swing_leg target
//! ```
//!
//! On the current tick `gait_out.legs[i].q_*` already encodes the
//! IK-solved joint angles for the swing trajectory at this tick — so
//! [`JointReference::from_controller_output`] is just a concentrating
//! wrapper that maps that into URDF sign convention and stages the
//! values for Position-PD + WBC consumption. Future ticks (1 step
//! ahead, etc.) are also covered by [`Self::predict_next_step`] which
//! uses [`crate::srbd_mpc::MpcSolution::predicted_body_states`] to
//! project the swing target more accurately when the body has
//! significant velocity.

use nalgebra::Vector3;

use crate::config::LegId;
use crate::controller::ControllerOutput;

/// Per-leg per-actuator joint reference in **URDF sign convention**
/// (= the same numbers that Position-PD writes into MuJoCo's
/// `position_target`).
///
/// `q[slot]` is `[hip, thigh, calf]` for FL/FR/RL/RR slots in the
/// canonical order (matches `LegId::ALL`).
#[derive(Clone, Copy, Debug, Default)]
pub struct JointReference {
    /// Joint position references per leg, URDF sign.
    pub q: [[f64; 3]; 4],
    /// Joint velocity references per leg, URDF sign. Set to zero by
    /// default; callers that finite-difference across ticks should
    /// fill these in (or use the analytical helper below).
    pub qd: [[f64; 3]; 4],
    /// Per-leg `is_stance` flag from the same controller output that
    /// produced `q`. Lets downstream tasks know which legs are in
    /// swing without re-deriving it from the gait clock.
    pub is_stance: [bool; 4],
}

impl JointReference {
    /// Build a [`JointReference`] from a single-tick
    /// [`ControllerOutput`]. The `q` field is the gait controller's
    /// IK output mapped through `joint_signs` to URDF sign.
    /// `qd` is left at zero (callers finite-difference if they want
    /// numeric q̇*).
    pub fn from_controller_output(
        out: &ControllerOutput,
        joint_signs: [[f64; 3]; 4],
    ) -> Self {
        let mut r = JointReference::default();
        for slot in 0..4 {
            let qs_ik = [
                out.legs[slot].q_hip,
                out.legs[slot].q_thigh,
                out.legs[slot].q_calf,
            ];
            for k in 0..3 {
                r.q[slot][k] = joint_signs[slot][k] * qs_ik[k];
            }
            r.is_stance[slot] = out.legs[slot].phase.is_stance;
            // qd left at 0 — caller can fill via finite-diff if
            // needed (the WbcPipeline does this internally).
        }
        r
    }

    /// Same as [`Self::from_controller_output`] but additionally fills
    /// `qd` by finite-differencing against the previous tick's `q`.
    /// Pass an all-zero `prev` and `dt = 0.0` on the first tick to
    /// suppress the spurious initial derivative.
    pub fn from_controller_output_with_fd(
        out: &ControllerOutput,
        joint_signs: [[f64; 3]; 4],
        prev_q_urdf: [[f64; 3]; 4],
        dt: f64,
    ) -> Self {
        let mut r = Self::from_controller_output(out, joint_signs);
        if dt > 1e-9 {
            for slot in 0..4 {
                for k in 0..3 {
                    r.qd[slot][k] = (r.q[slot][k] - prev_q_urdf[slot][k]) / dt;
                }
            }
        }
        r
    }
}

/// Convenience: index helper that maps `slot` (FL=0, FR=1, RL=2, RR=3)
/// to the corresponding [`LegId`] (so callers iterating both
/// representations don't have to maintain a parallel mapping).
pub fn leg_id_for_slot(slot: usize) -> LegId {
    LegId::ALL[slot]
}

// Suppress unused-warning when `Vector3` is only conditionally used.
#[allow(dead_code)]
fn _vector3_used(_: Vector3<f64>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LegId;
    use crate::controller::ControllerOutput;
    use crate::phase::PhaseState;
    use crate::LegOutput;

    fn make_controller_output() -> ControllerOutput {
        use crate::body_state::BodyState;
        use crate::footstep::Footstep;
        // Legs 0..3 with q values 0.1·k, 0.2·k, 0.3·k for k=hip/thigh/calf
        // and is_stance true for slots 0 and 2.
        let mk_leg = |slot: usize| LegOutput {
            leg: LegId::ALL[slot],
            hip_joint: String::new(),
            thigh_joint: String::new(),
            calf_joint: String::new(),
            q_hip: 0.1 * slot as f64,
            q_thigh: 0.2 * slot as f64,
            q_calf: 0.3 * slot as f64,
            foot_body: nalgebra::Vector3::zeros(),
            footstep: Footstep {
                lift_off: nalgebra::Vector3::zeros(),
                touch_down: nalgebra::Vector3::zeros(),
            },
            phase: PhaseState {
                leg: LegId::ALL[slot],
                cycle_position: 0.0,
                is_stance: slot % 2 == 0,
                sub_fraction: 0.0,
            },
            reachable: true,
        };
        ControllerOutput {
            legs: [mk_leg(0), mk_leg(1), mk_leg(2), mk_leg(3)],
            body_state: BodyState::default(),
        }
    }

    /// `from_controller_output` maps each leg's IK joint angles
    /// through `joint_signs` and copies the per-leg stance flag.
    #[test]
    fn from_controller_output_applies_signs_and_stance_flag() {
        let out = make_controller_output();
        // Use signs that flip thigh and calf (= namiashi convention).
        let signs = [[1.0, -1.0, -1.0]; 4];
        let r = JointReference::from_controller_output(&out, signs);
        for slot in 0..4 {
            // hip → +0.1·slot, thigh → -0.2·slot, calf → -0.3·slot
            assert!((r.q[slot][0] - 0.1 * slot as f64).abs() < 1e-12);
            assert!((r.q[slot][1] + 0.2 * slot as f64).abs() < 1e-12);
            assert!((r.q[slot][2] + 0.3 * slot as f64).abs() < 1e-12);
            assert_eq!(r.is_stance[slot], slot % 2 == 0);
            // qd zero by default.
            for k in 0..3 {
                assert_eq!(r.qd[slot][k], 0.0);
            }
        }
    }

    /// `from_controller_output_with_fd` populates `qd` via
    /// finite-difference; passing the previous tick's `q` and a
    /// non-zero `dt` reproduces `(q − q_prev) / dt` element-wise.
    #[test]
    fn finite_diff_qd_matches_dq_over_dt() {
        let out = make_controller_output();
        let signs = [[1.0, -1.0, -1.0]; 4];
        // Previous q: same as current minus 0.01 in every component
        // (= constant velocity 0.01/dt).
        let mut prev_q = [[0.0; 3]; 4];
        let r0 = JointReference::from_controller_output(&out, signs);
        for slot in 0..4 {
            for k in 0..3 {
                prev_q[slot][k] = r0.q[slot][k] - 0.01;
            }
        }
        let dt = 0.002;
        let r = JointReference::from_controller_output_with_fd(&out, signs, prev_q, dt);
        for slot in 0..4 {
            for k in 0..3 {
                let expected = 0.01 / dt;
                assert!(
                    (r.qd[slot][k] - expected).abs() < 1e-9,
                    "qd[{slot}][{k}] = {} vs expected {}",
                    r.qd[slot][k],
                    expected,
                );
            }
        }
    }
}
