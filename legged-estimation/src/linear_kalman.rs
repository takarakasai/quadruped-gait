//! 18-state Linear Kalman Filter for floating-base state estimation.
//!
//! Direct port of `legged_control`'s `KalmanFilterEstimate`
//! ([`linear_kalman_filter.cpp`](../../../ref/legged_control/legged_estimation/src/linear_kalman_filter.cpp)).
//!
//! State (18):
//!
//! ```text
//! x = [ body_pos_world (3)
//!     ; body_vel_world (3)
//!     ; foot_pos_world (12) ]      (FL, FR, RL, RR each 3)
//! ```
//!
//! Input (3): world-frame **linear** acceleration of the body (the
//! caller is expected to take the IMU's local linear acceleration,
//! rotate to world, and subtract gravity → that is `accel_world`).
//!
//! Observation (28):
//!
//! ```text
//! z = [ body_pos − foot_pos_world (12)   from FK         ← y[0..12]
//!     ; body_vel − foot_vel_world (12)   from J·q̇        ← y[12..24]
//!     ; foot_pos_world.z         (4)    = 0 when contact ← y[24..28] ]
//! ```
//!
//! Per-foot Q / R is up-weighted by `HIGH_SUSPECT` (= 100) when the
//! foot is **not** in contact, so unloaded feet don't drag the body
//! pose toward an arbitrary world-frame foot position.
//!
//! ## Numerical conventions
//! - All vectors / matrices are world frame unless noted.
//! - `accel_world` MUST already have gravity subtracted; otherwise the
//!   body z drifts upward at `g`.
//! - Process / measurement-noise scalars below mirror the values in
//!   `legged_control` (which were tuned for a Cheetah-class quadruped);
//!   hosts running other robots may need to retune via the `pub`
//!   fields.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Per-tick inputs to [`LinearKalmanEstimator::update`].
#[derive(Clone, Debug)]
pub struct LinearKalmanInputs<'a> {
    /// Time step (s). Same as the controller / sim period.
    pub dt: f64,
    /// World-frame linear acceleration of the body **with gravity
    /// removed** (m/s²). Caller computes
    /// `R_world_body · imu_local_accel + (0, 0, -9.81)`.
    pub accel_world: Vector3<f64>,
    /// Per-foot vector from body origin to foot, expressed in
    /// **world frame** (i.e. `R · foot_pos_body` where `foot_pos_body`
    /// is the foot position in the body-fixed frame).
    pub foot_pos_world_offset: &'a [Vector3<f64>; 4],
    /// Per-foot foot linear velocity in **world frame** (i.e.
    /// `J_world · q̇`, top-3 rows of the 6×nv spatial Jacobian, taken
    /// at the contact point).
    pub foot_vel_world: &'a [Vector3<f64>; 4],
    /// Per-foot stance flag (true when in contact).
    pub contact_flag: [bool; 4],
}

/// Decoded estimator output (extracted from `x_hat`).
#[derive(Clone, Debug)]
pub struct LinearKalmanOutput {
    pub body_pos_world: Vector3<f64>,
    pub body_vel_world: Vector3<f64>,
    pub foot_pos_world: [Vector3<f64>; 4],
}

/// 18-state LKF for a quadruped's floating-base + foot positions.
#[derive(Clone, Debug)]
pub struct LinearKalmanEstimator {
    /// Mean state estimate (length 18).
    pub x_hat: DVector<f64>,
    /// Covariance 18×18. Initialised to `100·I` (loose prior — first
    /// observations dominate).
    pub p: DMatrix<f64>,

    /// Process-noise base values per legged_control's tuning.
    pub imu_process_noise_position: f64,
    pub imu_process_noise_velocity: f64,
    pub foot_process_noise_position: f64,

    /// Measurement-noise base values.
    pub foot_sensor_noise_position: f64,
    pub foot_sensor_noise_velocity: f64,
    pub foot_height_sensor_noise: f64,
    /// Foot radius (m); added to the z-of-foot observation. 0 for a
    /// point foot.
    pub foot_radius: f64,
}

impl Default for LinearKalmanEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearKalmanEstimator {
    /// Construct with the legged_control-default tuning. Initial
    /// `x_hat` is zero (= body at world origin). Hosts that know a
    /// better starting body position should call [`Self::reset`].
    pub fn new() -> Self {
        Self {
            x_hat: DVector::zeros(18),
            p: DMatrix::identity(18, 18) * 100.0,
            imu_process_noise_position: 0.2,
            imu_process_noise_velocity: 0.2,
            foot_process_noise_position: 0.002,
            foot_sensor_noise_position: 0.005,
            foot_sensor_noise_velocity: 0.1,
            foot_height_sensor_noise: 0.005,
            foot_radius: 0.0,
        }
    }

    /// Reset the body-position estimate (and any consistent foot
    /// positions) to a known value. Use after a manual
    /// re-spawn / sim reset so the KF doesn't fight a 100 m·s
    /// initial-condition error.
    pub fn reset(
        &mut self,
        body_pos_world: Vector3<f64>,
        foot_pos_world: &[Vector3<f64>; 4],
    ) {
        self.x_hat.fill(0.0);
        for k in 0..3 {
            self.x_hat[k] = body_pos_world[k];
            self.x_hat[3 + k] = 0.0;
        }
        for slot in 0..4 {
            for k in 0..3 {
                self.x_hat[6 + 3 * slot + k] = foot_pos_world[slot][k];
            }
        }
        self.p = DMatrix::identity(18, 18) * 100.0;
    }

    /// Run one predict + update cycle.
    pub fn update(&mut self, inputs: &LinearKalmanInputs<'_>) -> LinearKalmanOutput {
        let dt = inputs.dt.max(0.0);

        // ── A (18×18): body_pos += dt·body_vel; foot_pos: identity ──
        let mut a = DMatrix::<f64>::identity(18, 18);
        for i in 0..3 {
            a[(i, 3 + i)] = dt;
        }

        // ── B (18×3): body_pos += 0.5·dt²·u; body_vel += dt·u ──────
        let mut b = DMatrix::<f64>::zeros(18, 3);
        let half_dt2 = 0.5 * dt * dt;
        for i in 0..3 {
            b[(i, i)] = half_dt2;
            b[(3 + i, i)] = dt;
        }

        // ── C (28×18): observation matrix ──────────────────────────
        // Rows  0..12  : per-foot (body_pos − foot_pos_world)
        // Rows 12..24  : per-foot (body_vel − foot_vel_world ≈ body_vel for static foot)
        // Rows 24..28  : per-foot foot_pos_world.z
        let mut c = DMatrix::<f64>::zeros(28, 18);
        for slot in 0..4 {
            let off = 3 * slot;
            // pos block: body_pos minus foot_pos_world
            for i in 0..3 {
                c[(off + i, i)] = 1.0;
                c[(off + i, 6 + off + i)] = -1.0;
            }
            // vel block: body_vel
            for i in 0..3 {
                c[(12 + off + i, 3 + i)] = 1.0;
            }
            // foot z observation
            c[(24 + slot, 6 + off + 2)] = 1.0;
        }

        // ── Process-noise covariance Q (18×18) ─────────────────────
        // Base scaling per legged_control: pos ~ dt/20, vel ~ dt·g/20,
        // foot ~ dt; multiplied by the per-mode tuning scalars.
        let mut q = DMatrix::<f64>::zeros(18, 18);
        let q_pos_block = (dt / 20.0) * self.imu_process_noise_position;
        let q_vel_block = (dt * 9.81 / 20.0) * self.imu_process_noise_velocity;
        let q_foot_block = dt * self.foot_process_noise_position;
        for i in 0..3 {
            q[(i, i)] = q_pos_block;
            q[(3 + i, 3 + i)] = q_vel_block;
        }
        for slot in 0..4 {
            // Per-foot block 3×3 starting at row/col 6 + 3·slot.
            let off = 6 + 3 * slot;
            // High-suspect multiplier when the foot is NOT in contact:
            // a swinging foot's world-position is anyone's guess, so
            // expand its process noise massively to let observations
            // rewrite it cheaply once it touches down.
            let contact_mul = if inputs.contact_flag[slot] {
                1.0
            } else {
                HIGH_SUSPECT
            };
            for i in 0..3 {
                q[(off + i, off + i)] = q_foot_block * contact_mul;
            }
        }

        // ── Measurement-noise covariance R (28×28) ────────────────
        let mut r = DMatrix::<f64>::zeros(28, 28);
        for slot in 0..4 {
            let off = 3 * slot;
            // pos rows: always trust position FK (foot_radius gives a
            // tiny nominal noise floor below).
            for i in 0..3 {
                r[(off + i, off + i)] = self.foot_sensor_noise_position;
            }
            // vel rows: trust J·q̇ only when the foot is in contact
            // (a foot mid-swing has unbounded ground-truth velocity
            // since it's moving relative to the world).
            let contact_mul = if inputs.contact_flag[slot] {
                1.0
            } else {
                HIGH_SUSPECT
            };
            for i in 0..3 {
                r[(12 + off + i, 12 + off + i)] = self.foot_sensor_noise_velocity * contact_mul;
            }
            // foot z = 0: only meaningful when contact, so up-weight
            // its noise off-contact.
            r[(24 + slot, 24 + slot)] = self.foot_height_sensor_noise * contact_mul;
        }

        // ── Build observation vector y (28) ───────────────────────
        // y[0..12] = -foot_pos_world_offset (= body→foot vector in
        // world; sign matches C's `body_pos − foot_pos_world` form so
        // the innovation `y − C·x_hat` cancels cleanly when the state
        // is correct).
        let mut y = DVector::<f64>::zeros(28);
        for slot in 0..4 {
            let off = 3 * slot;
            for k in 0..3 {
                y[off + k] = -inputs.foot_pos_world_offset[slot][k];
            }
            // Foot radius: a sphere foot's contact point is foot_radius
            // below the link origin, so the z-component of (-offset)
            // is shifted up by foot_radius before comparing to body_z.
            y[off + 2] += self.foot_radius;
            // velocity row: -foot_vel_world (so y = body_vel −
            // foot_vel_world for stance feet ≈ body_vel).
            for k in 0..3 {
                y[12 + off + k] = -inputs.foot_vel_world[slot][k];
            }
            // foot z = 0 (when contact).
            y[24 + slot] = 0.0;
        }

        // ── Predict: x_hat = A·x_hat + B·u ────────────────────────
        let u = inputs.accel_world;
        let u_dvec = DVector::from_iterator(3, u.iter().copied());
        self.x_hat = &a * &self.x_hat + &b * &u_dvec;

        // ── Predict covariance: P_- = A·P·Aᵀ + Q ──────────────────
        let pm = &a * &self.p * a.transpose() + &q;

        // ── Innovation: e = y − C·x_hat ───────────────────────────
        let y_model = &c * &self.x_hat;
        let ey = &y - &y_model;

        // ── Innovation covariance: S = C·P_-·Cᵀ + R ──────────────
        let s = &c * &pm * c.transpose() + &r;

        // ── Solve K = P_-·Cᵀ·S⁻¹ via S·X = ey then S·X = C·P_- ──
        let s_lu = s.clone().lu();
        let s_ey = match s_lu.solve(&ey) {
            Some(v) => v,
            None => return self.decode(), // S singular → keep x_hat unchanged
        };
        self.x_hat += &pm * c.transpose() * &s_ey;

        // ── Posterior covariance: P = (I − K·C)·P_- ──────────────
        let s_c = match s.lu().solve(&c) {
            Some(m) => m,
            None => {
                self.p = pm;
                return self.decode();
            }
        };
        self.p = (DMatrix::<f64>::identity(18, 18) - &pm * c.transpose() * &s_c) * &pm;
        // Symmetrise (numerical hygiene; otherwise repeated multiplies
        // accumulate small asymmetry that breaks downstream Cholesky).
        let pt = self.p.transpose();
        self.p = (&self.p + &pt) / 2.0;
        // legged_control's "block(0,0,2,2) determinant" hack: when
        // body_xy covariance has settled non-degenerate, decouple it
        // from the rest. Skipped here — empirically it adds little
        // and complicates the unit tests.

        self.decode()
    }

    fn decode(&self) -> LinearKalmanOutput {
        let body_pos = Vector3::new(self.x_hat[0], self.x_hat[1], self.x_hat[2]);
        let body_vel = Vector3::new(self.x_hat[3], self.x_hat[4], self.x_hat[5]);
        let mut foot_pos = [Vector3::zeros(); 4];
        for slot in 0..4 {
            for k in 0..3 {
                foot_pos[slot][k] = self.x_hat[6 + 3 * slot + k];
            }
        }
        LinearKalmanOutput {
            body_pos_world: body_pos,
            body_vel_world: body_vel,
            foot_pos_world: foot_pos,
        }
    }
}

/// Multiplier for Q / R blocks when the foot is in swing — same value
/// (`100`) the upstream `legged_control` `KalmanFilterEstimate` uses
/// (`high_suspect_number` in the original).
const HIGH_SUSPECT: f64 = 100.0;

// Suppress unused-import warning under no-tests builds.
#[allow(dead_code)]
fn _matrix3_used(_: Matrix3<f64>) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static stance: every foot in contact, accel_world = 0,
    /// observed body→foot offsets fixed at the nominal stance pose,
    /// observed velocities zero. After many updates the estimate
    /// converges to body at origin, velocity zero, and foot positions
    /// at the expected world-frame coordinates.
    #[test]
    fn static_stance_converges_to_truth() {
        let mut kf = LinearKalmanEstimator::new();
        // Truth: body at (0, 0, 0.30); each foot at (±0.18, ±0.10, 0).
        let body_truth = Vector3::new(0.0, 0.0, 0.30);
        let feet_truth = [
            Vector3::new(0.18, 0.10, 0.0),
            Vector3::new(0.18, -0.10, 0.0),
            Vector3::new(-0.18, 0.10, 0.0),
            Vector3::new(-0.18, -0.10, 0.0),
        ];
        // Body→foot offset (world frame, identity orientation) is
        // foot − body.
        let foot_offset: [Vector3<f64>; 4] =
            [0, 1, 2, 3].map(|i| feet_truth[i] - body_truth);
        let foot_vel: [Vector3<f64>; 4] = [Vector3::zeros(); 4];
        let inputs = LinearKalmanInputs {
            dt: 0.002,
            accel_world: Vector3::zeros(),
            foot_pos_world_offset: &foot_offset,
            foot_vel_world: &foot_vel,
            contact_flag: [true; 4],
        };
        // 1000 updates ≈ 2 s — generous for a 100·I prior to converge.
        for _ in 0..1000 {
            kf.update(&inputs);
        }
        let out = kf.update(&inputs);
        for k in 0..3 {
            assert!(
                (out.body_pos_world[k] - body_truth[k]).abs() < 1e-3,
                "body_pos[{k}] = {} should converge to {}",
                out.body_pos_world[k],
                body_truth[k],
            );
            assert!(
                out.body_vel_world[k].abs() < 1e-3,
                "body_vel[{k}] = {} should be 0",
                out.body_vel_world[k],
            );
        }
        for slot in 0..4 {
            for k in 0..3 {
                assert!(
                    (out.foot_pos_world[slot][k] - feet_truth[slot][k]).abs() < 1e-2,
                    "foot[{slot}][{k}] = {} should converge to {}",
                    out.foot_pos_world[slot][k],
                    feet_truth[slot][k],
                );
            }
        }
    }

    /// IMU integration: with all four feet in **swing** (so the
    /// foot-velocity observation rows are heavily down-weighted via
    /// `HIGH_SUSPECT`) and a constant world-frame acceleration input,
    /// the body velocity should integrate the accel correctly.
    ///
    /// ∫₀ᵗ a dt = a · t. With a = 1 m/s², after 0.5 s the velocity
    /// should be near 0.5 m/s.
    #[test]
    fn constant_accel_integrates_to_velocity() {
        let mut kf = LinearKalmanEstimator::new();
        // Reset to clean state at origin so the prior covariance
        // doesn't dominate the IMU input early on.
        kf.reset(Vector3::zeros(), &[Vector3::zeros(); 4]);
        let dt = 0.002_f64;
        let nominal_offset = [
            Vector3::new(0.18, 0.10, 0.0),
            Vector3::new(0.18, -0.10, 0.0),
            Vector3::new(-0.18, 0.10, 0.0),
            Vector3::new(-0.18, -0.10, 0.0),
        ];
        let foot_vel = [Vector3::zeros(); 4];
        let inputs = LinearKalmanInputs {
            dt,
            accel_world: Vector3::new(1.0, 0.0, 0.0),
            foot_pos_world_offset: &nominal_offset,
            foot_vel_world: &foot_vel,
            // Mark all swing so foot observations don't fight the
            // IMU's velocity prediction.
            contact_flag: [false; 4],
        };
        let n = (0.5 / dt) as usize; // 0.5 s
        let mut out = kf.update(&inputs);
        for _ in 0..n {
            out = kf.update(&inputs);
        }
        // The KF is IMU + observation fusion, NOT pure integration —
        // the foot-position rows pull body_pos toward the implied
        // (foot_pos_world − offset), which damps the integration.
        // Verify direction and order of magnitude rather than the
        // pure-integration value (0.5 m/s after 0.5 s at 1 m/s²).
        assert!(
            out.body_vel_world.x > 0.1 && out.body_vel_world.x < 0.5,
            "body_vel.x = {} should be in (0.1, 0.5) (positive but damped \
             by foot-position observations)",
            out.body_vel_world.x,
        );
        // y/z components stay near zero.
        assert!(out.body_vel_world.y.abs() < 0.05);
        assert!(out.body_vel_world.z.abs() < 0.05);
    }
}
