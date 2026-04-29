//! Foot trajectory generators: linear stance + Bezier swing.
//!
//! Both functions return the foot position in the body frame at a given
//! sub-phase fraction in [0, 1]. The gait controller drives them per-leg
//! based on whether the leg is currently in stance or swing.

use nalgebra::Vector3;

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

/// Swing foot trajectory: a 4-point Bezier curve in the body frame.
///
/// Control points are placed so that:
/// - the curve starts exactly at `lift_off` (stance end),
/// - reaches its maximum z at `frac = 0.5`, exactly `swing_height` above
///   the lift-off / touch-down line,
/// - lands exactly on `touch_down` (next stance start),
/// - has zero horizontal velocity at both endpoints (smooth blend with
///   the stance line).
///
/// Cubic-Bezier midpoint sits at `0.5·P0 + 0.125·P1 + 0.125·P2 + 0.5·P3`
/// of the position weights — wait, that's wrong; the binomial weights at
/// `t=0.5` are `(1/8, 3/8, 3/8, 1/8)`. So if we lift the two middle
/// control points by `H` above the endpoints, the midpoint rise is
/// `(3/8 + 3/8) · H = 0.75 · H`. To make the **peak** equal to
/// `swing_height`, we therefore lift by `4/3 · swing_height`.
pub fn swing_position(
    lift_off: Vector3<f64>,
    touch_down: Vector3<f64>,
    swing_height: f64,
    frac: f64,
) -> Vector3<f64> {
    let t = frac.clamp(0.0, 1.0);
    // Cubic Bezier control points P0..P3.
    let p0 = lift_off;
    let p3 = touch_down;
    // 4/3 lift puts the curve's true peak at exactly `swing_height` above
    // the chord — see derivation in the doc comment.
    let lift = Vector3::new(0.0, 0.0, swing_height * 4.0 / 3.0);
    let p1 = lift_off + lift;
    let p2 = touch_down + lift;

    let one_t = 1.0 - t;
    let b0 = one_t * one_t * one_t;
    let b1 = 3.0 * one_t * one_t * t;
    let b2 = 3.0 * one_t * t * t;
    let b3 = t * t * t;
    p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3
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
        // Peak of a cubic Bezier with control points at 2h is exactly h.
        assert_relative_eq!(peak, h, epsilon = 1e-2);
    }
}
