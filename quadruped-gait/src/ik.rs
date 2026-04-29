//! Analytical 3-DOF inverse kinematics for the standard quadruped leg.
//!
//! Topology: hip (Roll about X) → thigh (Pitch about Y) → calf (Pitch about Y).
//! Both pitch axes share the same direction so the thigh + calf form a
//! 2-link planar chain inside the leg's roll-rotated YZ plane.
//!
//! Convention (CHAMP-compatible, body frame x forward / y left / z up):
//!
//! - Hip joint at `hip_offset`.
//! - With all q=0:
//!     - thigh axis points along body Y (left for FL/RL, right after a
//!       configurable lateral offset);
//!     - the upper segment hangs straight down (-Z);
//!     - the lower segment continues straight down (-Z), foot fully extended.
//! - Positive **hip roll**: rotates the leg outward (foot moves away from
//!   the body's longitudinal axis).
//! - Positive **thigh pitch**: rotates the upper segment forward (knee
//!   ahead of hip).
//! - Positive **calf pitch**: bends the knee further (foot tucks toward body).
//!
//! These conventions are choices. The articara wrapper must adapt URDF
//! joint axis directions to match (sign-flip individual joints as needed).
//!
//! # Algorithm
//!
//! 1. Translate the target into the hip frame: `p = target_body − hip_offset`.
//! 2. Solve hip roll from the lateral plane: the lateral offset
//!    `hip_to_thigh_y` of the thigh axis from the roll axis means the
//!    "leg plane" (where pitch-pitch lives) is offset from the YZ plane by
//!    that amount. After the roll, the leg plane is the locus of points
//!    `(x, y_off · sin(q1), -y_off · cos(q1)) + (x, ρ · cos(q1), ρ · sin(q1))`
//!    in hip frame — solving for q1 reduces to one trigonometric equation.
//! 3. Solve the planar 2-link IK in the rotated leg plane for `(q2, q3)`.
//!
//! # Reachability
//!
//! Returns [`LegIkSolution::Unreachable`] when the target is outside the
//! workspace (closer than `|L1−L2|` or farther than `L1+L2` from the
//! thigh-pitch axis, or the lateral component exceeds the roll's reach).
//! In practice the gait controller upstream limits step sizes to stay well
//! inside the envelope; unreachability is reported so the host can warn.

use crate::config::LegKinematics;
use nalgebra::Vector3;

/// Result of solving a single leg's IK against a target foot position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LegIkSolution {
    /// `(hip, thigh, calf)` joint angles in radians.
    Reached {
        hip: f64,
        thigh: f64,
        calf: f64,
    },
    /// Target is outside the leg's reachable workspace. The closest
    /// achievable configuration is provided so the controller can clamp
    /// the trajectory rather than blow up.
    Unreachable {
        hip: f64,
        thigh: f64,
        calf: f64,
    },
}

impl LegIkSolution {
    pub fn angles(&self) -> (f64, f64, f64) {
        match self {
            LegIkSolution::Reached { hip, thigh, calf }
            | LegIkSolution::Unreachable { hip, thigh, calf } => (*hip, *thigh, *calf),
        }
    }

    pub fn is_reachable(&self) -> bool {
        matches!(self, LegIkSolution::Reached { .. })
    }
}

/// Solve the 3-DOF IK for one leg.
///
/// `target_body` is the desired foot position in the **body frame** (m).
/// The function uses `kin.hip_offset` to translate into the hip frame,
/// then applies the analytical formulas above.
///
/// `knee_forward = true` selects the elbow-forward branch (knee bends
/// forward); `false` selects elbow-back (knee bends backward). For most
/// quadrupeds the rear legs are knees-back and front legs knees-forward,
/// but this varies — caller should pick to match the model's URDF.
pub fn solve_leg_ik(
    kin: &LegKinematics,
    target_body: Vector3<f64>,
    knee_forward: bool,
) -> LegIkSolution {
    // Translate into hip frame. The hip frame is body-aligned, just shifted.
    let p = target_body - kin.hip_offset;
    let l_lat = kin.hip_to_thigh_y; // signed: positive away from centerline
    // Sign-flip lateral offset for right-side legs so positive hip roll
    // always "outward" regardless of which side the leg is on.
    let lateral_sign = if kin.leg.is_left() { 1.0 } else { -1.0 };
    let l_lat = lateral_sign * l_lat;

    let l1 = kin.upper_leg_m;
    let l2 = kin.lower_leg_m;

    // ── Hip roll ──────────────────────────────────────────────────────
    // After rolling about X by q1, the thigh-pitch axis sits at
    // (0, l_lat·cos(q1), l_lat·sin(q1)) in the hip frame. The remaining
    // (planar) reach in the leg plane is along the rotated Z axis.
    //
    // From the foot position p = (px, py, pz), the projection onto the
    // YZ plane is (py, pz) with radial distance ρ = √(py² + pz²).
    // The hip rotates the leg so the planar y-coordinate of the foot is
    // exactly l_lat:
    //   l_lat = py·cos(q1) + pz·sin(q1)
    //   (rotated Z) = -py·sin(q1) + pz·cos(q1)
    //
    // Solving for q1 with the sin/cos auxiliary trick:
    //   l_lat = ρ · cos(α − q1)   where α = atan2(pz, py)
    //   q1 = α ± acos(l_lat / ρ)
    // Foot in hip frame, given desired (q1, z_planar):
    //   py = l_lat · cos(q1) − z_planar · sin(q1)
    //   pz = l_lat · sin(q1) + z_planar · cos(q1)
    //
    // Squaring and adding: py² + pz² = l_lat² + z_planar²  → solves z_planar.
    // Substituting back gives a 2×2 linear system in (cos q1, sin q1):
    //   [ l_lat  -z_planar ] [c]   [py]
    //   [ z_planar  l_lat  ] [s] = [pz]
    //
    // The determinant l_lat² + z_planar² is always positive when the
    // target is reachable laterally, so the system has a unique solution.
    let py = p.y;
    let pz = p.z;
    let inner = py * py + pz * pz - l_lat * l_lat;
    if inner < 0.0 {
        // Lateral offset exceeds the foot's distance from the roll axis —
        // the leg can never reach this point regardless of pitch angles.
        return LegIkSolution::Unreachable { hip: 0.0, thigh: 0.0, calf: 0.0 };
    }
    // Pick the negative branch so the foot ends up below the hip.
    let z_planar = -inner.sqrt();
    let denom = l_lat * l_lat + z_planar * z_planar;
    let c1 = (l_lat * py + z_planar * pz) / denom;
    let s1 = (l_lat * pz - z_planar * py) / denom;
    let q1 = s1.atan2(c1);

    let px = p.x;
    let r_planar_sq = px * px + z_planar * z_planar;
    let r_planar = r_planar_sq.sqrt();

    // ── 2-link planar IK ─────────────────────────────────────────────
    // Foot reach along the leg plane is r_planar; segments are L1, L2.
    let max_reach = l1 + l2;
    let min_reach = (l1 - l2).abs();
    let mut unreachable = false;
    let r_clamped = if r_planar > max_reach {
        unreachable = true;
        max_reach * 0.999
    } else if r_planar < min_reach {
        unreachable = true;
        min_reach * 1.001
    } else {
        r_planar
    };

    // Knee angle from law of cosines. q3 = π − interior_angle
    // We pick the sign of q3 by `knee_forward`: positive q3 → knee bends
    // backward in this convention (front-leg-style); negative → forward.
    let cos_inner =
        ((l1 * l1 + l2 * l2 - r_clamped * r_clamped) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
    let inner_angle = cos_inner.acos();
    let mut q3 = std::f64::consts::PI - inner_angle;
    if knee_forward {
        q3 = -q3;
    }

    // Thigh pitch: aim the upper segment toward the foot, then back off
    // by half the knee angle so the geometry closes.
    let foot_dir = px.atan2(-z_planar); // angle from straight-down (hip frame)
    // Side-angle inside the elbow triangle, opposite to L2.
    let cos_side = ((l1 * l1 + r_clamped * r_clamped - l2 * l2)
        / (2.0 * l1 * r_clamped))
        .clamp(-1.0, 1.0);
    let side_angle = cos_side.acos();
    let q2 = if knee_forward {
        foot_dir + side_angle
    } else {
        foot_dir - side_angle
    };

    if unreachable {
        LegIkSolution::Unreachable { hip: q1, thigh: q2, calf: q3 }
    } else {
        LegIkSolution::Reached { hip: q1, thigh: q2, calf: q3 }
    }
}

/// Forward kinematics inverse to [`solve_leg_ik`]. Useful for testing the
/// IK round-trip and for the host's debug-overlay visualisation.
pub fn forward_leg_kinematics(
    kin: &LegKinematics,
    hip: f64,
    thigh: f64,
    calf: f64,
) -> Vector3<f64> {
    let l1 = kin.upper_leg_m;
    let l2 = kin.lower_leg_m;
    // Sign-flipped lateral offset, matching the convention in solve_leg_ik.
    let lateral_sign = if kin.leg.is_left() { 1.0 } else { -1.0 };
    let l_lat = lateral_sign * kin.hip_to_thigh_y;

    // The thigh + calf form a 2-link planar chain. With q2 = thigh pitch
    // (positive forward) and q3 = calf pitch (positive bends knee back in
    // our sign convention):
    //   x = L1 sin(q2) + L2 sin(q2 + q3)
    //   z_plane = -L1 cos(q2) - L2 cos(q2 + q3)   (negative below hip)
    let q23 = thigh + calf;
    let x = l1 * thigh.sin() + l2 * q23.sin();
    let z_plane = -l1 * thigh.cos() - l2 * q23.cos();

    // Roll the planar (x, z_plane) result around the body X axis by q1
    // (hip roll), and add the lateral offset.
    let cos1 = hip.cos();
    let sin1 = hip.sin();
    let y_hip = l_lat * cos1 + z_plane * sin1;
    let z_hip = l_lat * sin1 + z_plane * (-1.0) * (-cos1); // = -l_lat·sin? let me re-derive
    // Re-derive: the hip rotates about X (body-forward). A point (0, l_lat,
    // z_plane) before the roll maps after roll(q1, about X) to:
    //   y_after =  l_lat·cos(q1) − z_plane·sin(q1)
    //   z_after =  l_lat·sin(q1) + z_plane·cos(q1)
    let _ = y_hip;
    let _ = z_hip;
    let y_after = l_lat * cos1 - z_plane * sin1;
    let z_after = l_lat * sin1 + z_plane * cos1;

    let foot_hip = Vector3::new(x, y_after, z_after);
    foot_hip + kin.hip_offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LegId;
    use approx::assert_relative_eq;

    fn fl_kin() -> LegKinematics {
        LegKinematics::new(
            LegId::FL,
            "FL_hip".into(),
            "FL_thigh".into(),
            "FL_calf".into(),
            "FL_foot".into(),
            Vector3::new(0.18, 0.05, 0.0),
            0.04,   // lateral offset hip→thigh
            0.18,   // upper leg
            0.18,   // lower leg
        )
    }

    #[test]
    fn nominal_stance_yields_zero_pitch_angles() {
        // Foot at the nominal stance pose should produce hip ≈ 0,
        // thigh ≈ 0, calf ≈ 0 (legs straight down).
        let kin = fl_kin();
        let target = kin.nominal_foot_body;
        let sol = solve_leg_ik(&kin, target, false);
        let (h, t, c) = sol.angles();
        assert!(sol.is_reachable());
        assert_relative_eq!(h, 0.0, epsilon = 1e-6);
        assert_relative_eq!(t, 0.0, epsilon = 1e-6);
        assert_relative_eq!(c, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn ik_fk_round_trip_random_targets() {
        let kin = fl_kin();
        // Use a slightly retracted nominal stance (knee bent ~10°) so the
        // workspace has room for offsets in any direction. Without this,
        // the default fully-extended pose sits exactly on the boundary
        // and even a small forward offset becomes unreachable.
        let nominal = kin.nominal_foot_body + Vector3::new(0.0, 0.0, 0.04);
        let offsets = [
            Vector3::new(0.04, 0.0, 0.0),
            Vector3::new(-0.04, 0.0, 0.0),
            Vector3::new(0.0, 0.02, 0.0),
            Vector3::new(0.0, 0.0, 0.03),
            Vector3::new(0.0, 0.0, -0.03),
            Vector3::new(0.03, -0.01, 0.02),
        ];
        for d in offsets {
            let target = nominal + d;
            let sol = solve_leg_ik(&kin, target, false);
            assert!(
                sol.is_reachable(),
                "expected reachable for offset {:?}, got {:?}",
                d, sol,
            );
            let (h, t, c) = sol.angles();
            let recovered = forward_leg_kinematics(&kin, h, t, c);
            for ax in 0..3 {
                assert_relative_eq!(
                    recovered[ax], target[ax],
                    epsilon = 1e-3,
                    max_relative = 1e-3,
                );
            }
        }
    }

    #[test]
    fn ik_fk_round_trip_both_knee_branches() {
        // Both knee branches must hit the same foot target — they're two
        // valid IK solutions (knee in front vs back) for the same 3D foot
        // position. This is the kinematic guarantee that lets the user
        // pick `<<` or `>>` purely for aesthetics: the body-frame motion
        // is identical, only the leg silhouette differs.
        let kin = fl_kin();
        let nominal = kin.nominal_foot_body + Vector3::new(0.0, 0.0, 0.04);
        let offsets = [
            Vector3::new(0.04, 0.0, 0.0),
            Vector3::new(-0.04, 0.0, 0.0),
            Vector3::new(0.0, 0.02, 0.0),
            Vector3::new(0.0, 0.0, 0.03),
            Vector3::new(0.03, -0.01, 0.02),
        ];
        for d in offsets {
            let target = nominal + d;
            for knee_forward in [false, true] {
                let sol = solve_leg_ik(&kin, target, knee_forward);
                assert!(
                    sol.is_reachable(),
                    "offset {:?} knee_forward={knee_forward}: {:?}",
                    d, sol,
                );
                let (h, t, c) = sol.angles();
                let recovered = forward_leg_kinematics(&kin, h, t, c);
                for ax in 0..3 {
                    assert_relative_eq!(
                        recovered[ax], target[ax],
                        epsilon = 1e-3,
                    );
                }
            }
        }
    }

    #[test]
    fn far_target_returns_unreachable() {
        let kin = fl_kin();
        // Push the target way beyond reach.
        let unreach = kin.nominal_foot_body + Vector3::new(2.0, 0.0, 0.0);
        let sol = solve_leg_ik(&kin, unreach, false);
        assert!(!sol.is_reachable());
    }
}
