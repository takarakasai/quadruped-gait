//! Foot trajectory generators: linear stance + smoothstep / sin² swing.
//!
//! Both functions return the foot position in the body frame at a given
//! sub-phase fraction in [0, 1]. The gait controller drives them per-leg
//! based on whether the leg is currently in stance or swing.
//!
//! # Why not a single cubic Bezier for swing?
//!
//! The first cut of [`swing_position`] used a cubic Bezier with control
//! points `[lift_off, lift_off + 4/3·h·ẑ, touch_down + 4/3·h·ẑ,
//! touch_down]`. That guaranteed the right peak height **and** zero
//! xy-velocity at the endpoints (so stance↔swing handoffs were smooth in
//! the horizontal plane), but it landed with a non-zero **vertical**
//! velocity:
//!
//! ```text
//! B'(1) = 3·(P3 − P2) = (0, 0, −4·h)
//! ```
//!
//! Per swing cycle that's ~0.8 m/s downward at touchdown for the
//! defaults (h = 4 cm, T_swing = 0.2 s). Real-world: trunk visibly
//! bobs every cycle, especially when turning or strafing — the
//! asymmetric stride amplitudes between inner / outer legs amplify the
//! per-foot impact unevenly and the body bounces.
//!
//! The replacement decouples xy from z so we can pick smooth-end
//! profiles for both:
//!
//! - **xy**: smoothstep `S(t) = 3t² − 2t³` (zero velocity at 0 and 1)
//! - **z**:  raised-cosine bump `B(t) = sin²(π·t) · h` (zero velocity
//!   at 0 and 1, peak `h` at t = 0.5)
//!
//! Touchdown vertical velocity is now *exactly* zero. The bump is
//! C¹-continuous to the stance line at both ends, so the handoff
//! still introduces no torque transient.

use nalgebra::Vector3;

#[cfg(test)] // test-only reference implementation / helper
/// Stance foot trajectory: a straight line from `lift_off` to `touch_down`.
///
/// `frac` is the leg's stance sub-phase, 0 → just hit the ground,
/// 1 → about to lift off again.
pub fn stance_position(
    lift_off: Vector3<f64>,
    touch_down: Vector3<f64>,
    frac: f64,
) -> Vector3<f64> {
    let f = frac.clamp(0.0, 1.0);
    lift_off * (1.0 - f) + touch_down * f
}

/// Swing foot trajectory: smoothstep in xy, sin² bump in z.
///
/// Both axes start and end with zero velocity, so handing over to
/// [`stance_position`] generates no impulsive torque on the leg. The
/// **vertical** zero-velocity property is the change from the
/// historical cubic-Bezier curve and is what kills the trunk-bobbing
/// previously seen on touchdown (see module docs).
///
/// - `frac = 0` → returns `lift_off` exactly.
/// - `frac = 1` → returns `touch_down` exactly.
/// - `frac = 0.5` → peak z is `swing_height` above the chord.
pub fn swing_position(
    lift_off: Vector3<f64>,
    touch_down: Vector3<f64>,
    swing_height: f64,
    frac: f64,
) -> Vector3<f64> {
    let t = frac.clamp(0.0, 1.0);

    // ── xy: smoothstep S(t) = 3t² − 2t³ ───────────────────────────────
    // Maps [0, 1] → [0, 1] with S(0) = S(1) = 0 derivative. Linear in
    // value at t = 0.5 (S(0.5) = 0.5), so the foot's mean horizontal
    // speed equals the chord length / swing duration as expected.
    let s = (3.0 - 2.0 * t) * t * t;
    let xy_chord = touch_down - lift_off;
    let mut p = lift_off + xy_chord * s;
    // The xy formula above also moves z linearly between lift_off.z
    // and touch_down.z (it scales the whole chord vector). That part
    // is fine — it's the chord baseline. We add the swing bump on top.

    // ── z bump: sin²(π·t) · swing_height ───────────────────────────────
    // sin² has zero derivative at t = 0 and t = 1 → soft landing AND
    // soft lift-off. Peak `swing_height` at t = 0.5. C¹-continuous to
    // the stance line on both ends.
    let bump = (std::f64::consts::PI * t).sin().powi(2) * swing_height;
    p.z += bump;
    p
}

/// Planned world-frame vertical foot velocity at sub-phase `frac` of a
/// swing whose total duration is `swing_duration_s` seconds. The base
/// curve is the analytical derivative of [`swing_position`]'s z bump
/// (`sin²(π·t)·h`), normalized to the swing's real-time duration:
///
/// ```text
/// dz/dt = h · π · sin(2π·t) / T_swing
/// ```
///
/// `lift_off_vz` / `touch_down_vz` add linear blends from the endpoints
/// so the caller can switch between the articara-native zero-boundary
/// profile (both = 0, default) and legged_control's nonzero-boundary
/// profile (lift_off_vz ≈ +0.05, touch_down_vz ≈ −0.10) without
/// touching [`swing_position`]'s position curve. The blend uses a
/// `(1−t)` / `t` ramp so each endpoint exactly matches its requested
/// boundary value and the sin² peak is left intact at the midpoint.
///
/// Returned in m/s. Assumes the body frame's z-axis is aligned with the
/// world's vertical, which is the small-angle linearisation regime
/// the full-centroidal MPC operates under.
pub fn swing_vz_world(
    swing_height: f64,
    frac: f64,
    swing_duration_s: f64,
    lift_off_vz: f64,
    touch_down_vz: f64,
) -> f64 {
    let t = frac.clamp(0.0, 1.0);
    let t_swing = swing_duration_s.max(1e-6);
    let two_pi_t = 2.0 * std::f64::consts::PI * t;
    let bump_dot = swing_height * std::f64::consts::PI * two_pi_t.sin() / t_swing;
    let boundary = (1.0 - t) * lift_off_vz + t * touch_down_vz;
    bump_dot + boundary
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stance_endpoints_exact() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(1.0, 2.0, -0.3);
        for k in 0..2 {
            let f = k as f64;
            let p = stance_position(a, b, f);
            let expected = if k == 0 { a } else { b };
            for i in 0..3 {
                assert_relative_eq!(p[i], expected[i], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn swing_endpoints_exact() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.1, 0.0, 0.0);
        let p_start = swing_position(a, b, 0.05, 0.0);
        let p_end = swing_position(a, b, 0.05, 1.0);
        for i in 0..3 {
            assert_relative_eq!(p_start[i], a[i], epsilon = 1e-9);
            assert_relative_eq!(p_end[i], b[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn swing_peak_above_endpoints() {
        // At t = 0.5 the foot must be visibly higher than start/end.
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.1, 0.0, 0.0);
        let h = 0.04;
        let p_mid = swing_position(a, b, h, 0.5);
        // Sweep many points to find the peak.
        let mut peak = f64::NEG_INFINITY;
        for k in 0..=100 {
            let t = k as f64 / 100.0;
            let p = swing_position(a, b, h, t);
            if p.z > peak {
                peak = p.z;
            }
        }
        assert!(p_mid.z > 0.0, "swing midpoint should rise above ground");
        // Peak should be exactly `h` (sin² at t=0.5 is 1).
        assert_relative_eq!(peak, h, epsilon = 1e-3);
        assert_relative_eq!(p_mid.z, h, epsilon = 1e-9);
    }

    /// Critical regression: the swing curve must hit touchdown with
    /// **zero** vertical velocity. Prior cubic-Bezier curve had ~−4·h
    /// per unit time, which translated to ~0.8 m/s downward at default
    /// settings — that's the source of trunk bobbing during turn /
    /// strafe. The replacement uses sin² so dz/dt vanishes at both
    /// ends.
    #[test]
    fn swing_zero_vertical_velocity_at_endpoints() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.10, 0.02, 0.0);
        let h = 0.04;
        let eps = 1e-5;
        // Numerical derivative just inside each endpoint.
        let p0 = swing_position(a, b, h, 0.0);
        let p_eps = swing_position(a, b, h, eps);
        let p1 = swing_position(a, b, h, 1.0);
        let p_one_minus_eps = swing_position(a, b, h, 1.0 - eps);

        let vz_lift = (p_eps.z - p0.z) / eps;
        let vz_land = (p1.z - p_one_minus_eps.z) / eps;
        // sin²'(π·0) = 0 and sin²'(π·1) = 0 → derivative is exactly
        // zero in the limit, but our finite-difference probe is O(eps).
        assert!(
            vz_lift.abs() < 1e-3,
            "lift-off vertical velocity should be ~0, got {vz_lift}"
        );
        assert!(
            vz_land.abs() < 1e-3,
            "touchdown vertical velocity should be ~0, got {vz_land}"
        );
    }

    /// Horizontal velocity at the endpoints must also be zero so the
    /// stance↔swing handoff is C¹ in xy too.
    #[test]
    fn swing_zero_horizontal_velocity_at_endpoints() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.10, 0.02, 0.0);
        let h = 0.04;
        let eps = 1e-5;
        let p0 = swing_position(a, b, h, 0.0);
        let p_eps = swing_position(a, b, h, eps);
        let p1 = swing_position(a, b, h, 1.0);
        let p_one_minus_eps = swing_position(a, b, h, 1.0 - eps);

        let vxy_lift = ((p_eps - p0) / eps).xy().norm();
        let vxy_land = ((p1 - p_one_minus_eps) / eps).xy().norm();
        // smoothstep S'(0) = S'(1) = 0 exactly.
        assert!(vxy_lift < 1e-3, "lift-off xy velocity should be ~0, got {vxy_lift}");
        assert!(vxy_land < 1e-3, "touchdown xy velocity should be ~0, got {vxy_land}");
    }

    /// `swing_vz_world` must agree with a finite-difference of
    /// `swing_position`'s z-component when boundary velocities are zero.
    /// This is the consistency check between the position curve and the
    /// reference velocity the MPC's swing-leg constraint receives.
    #[test]
    fn swing_vz_world_matches_position_derivative() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.10, 0.02, 0.0);
        let h = 0.04;
        let t_swing = 0.25;
        let eps = 1e-5;
        for k in 1..10 {
            let t = k as f64 / 10.0;
            let p_minus = swing_position(a, b, h, t - eps);
            let p_plus = swing_position(a, b, h, t + eps);
            // Finite-difference dz/dfrac, then divide by T_swing to get
            // dz/dt as a real-time velocity.
            let dz_dfrac = (p_plus.z - p_minus.z) / (2.0 * eps);
            let vz_fd = dz_dfrac / t_swing;
            let vz_an = swing_vz_world(h, t, t_swing, 0.0, 0.0);
            assert!(
                (vz_fd - vz_an).abs() < 1e-4,
                "swing_vz_world disagrees with d/dt(swing_position.z) at t={t}: fd={vz_fd}, an={vz_an}"
            );
        }
    }

    /// Boundary-velocity blend: at frac=0 and frac=1 the analytical
    /// vz must equal lift_off_vz / touch_down_vz exactly (sin² term is
    /// zero at both endpoints).
    #[test]
    fn swing_vz_world_boundary_blend() {
        let h = 0.04;
        let t_swing = 0.25;
        let vz0 = swing_vz_world(h, 0.0, t_swing, 0.05, -0.10);
        let vz1 = swing_vz_world(h, 1.0, t_swing, 0.05, -0.10);
        assert_relative_eq!(vz0, 0.05, epsilon = 1e-9);
        assert_relative_eq!(vz1, -0.10, epsilon = 1e-9);
    }

    /// Swing curve should monotonically traverse the chord in xy
    /// (smoothstep is monotone non-decreasing), so the foot doesn't
    /// loop back during swing.
    #[test]
    fn swing_xy_monotone() {
        let a = Vector3::new(0.0, 0.0, 0.0);
        let b = Vector3::new(0.10, 0.02, 0.0);
        let h = 0.04;
        let mut prev_proj = f64::NEG_INFINITY;
        let chord = b - a;
        let chord_len = chord.norm();
        for k in 0..=100 {
            let t = k as f64 / 100.0;
            let p = swing_position(a, b, h, t);
            let proj = (p - a).xy().dot(&chord.xy()) / chord_len;
            assert!(
                proj >= prev_proj - 1e-9,
                "xy projection must be monotone non-decreasing (t={t}, prev={prev_proj}, now={proj})"
            );
            prev_proj = proj;
        }
    }
}
