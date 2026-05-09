//! Centroidal-momentum SRBD MPC.
//!
//! Smaller-scope sibling of [`crate::srbd_mpc`]: same convex MPC
//! family (Di Carlo-style 12-dim QP), but the state is expressed in
//! **centroidal** coordinates instead of body-root coordinates.
//! Mirrors the `centroidalModelType = 1` path of `legged_control`'s
//! ocs2 stack.
//!
//! ## Why a new module instead of patching `SrbdMpc`?
//!
//! The body-root SRBD assumes the body's CoM is at the body root.
//! For most quadrupeds in the Cheetah-3 lineage this is approximately
//! true (the trunk's inertial origin is at the URDF root and the legs
//! are mass-balanced). For namiashi the trunk_inertia link is offset
//! ~5 mm in +y from the root, and the resulting gravity-induced
//! rolling moment is mishandled by the body-root formulation
//! (documented in `tests/integration_walk.rs::diag_constrain_pose_axis_swap`).
//! Patching `SrbdMpc` to "know about" the offset breaks the QP's self-
//! consistency between linear (`v̇ = Σf/m`) and angular dynamics
//! (`α = I⁻¹·Σr×f`). The clean fix is to *change the state space* so
//! the linear "velocity" is the CoM velocity by definition — that's
//! what this module does.
//!
//! ## State space (12-dim, type-1 in legged_control)
//!
//! ```text
//! x = [ ḣ_lin / m  (3)   = v_CoM, m/s, world frame
//!       ḣ_ang / m  (3)   = angular momentum / mass, m²/s, world frame
//!       base_pos   (3)   = body root position in world frame, m
//!       euler_ZYX  (3)   = base orientation, [roll(x), pitch(y), yaw(z)], rad
//!     ]
//! ```
//!
//! Note: state stores **angular momentum / mass** rather than angular
//! velocity, matching ocs2_centroidal_model's convention. Angular
//! velocity recovers via `ω = I_centroidal⁻¹ · h_ang`, where
//! `I_centroidal` is the centroidal angular inertia (constant under
//! type-1 SRBD = constant CMM evaluated at a nominal pose).
//!
//! ## Input space (12-dim)
//!
//! ```text
//! u = [ F_FL, F_FR, F_RL, F_RR ]   each 3-vector in world frame, N
//! ```
//!
//! Same as `SrbdMpc::Input` — the GRF applied at each foot.
//!
//! ## Continuous-time dynamics
//!
//! For type-1 SRBD (constant centroidal inertia, foot positions
//! sampled per-step from outside):
//!
//! ```text
//! d/dt (h_lin/m)   = (Σ F)/m + g
//! d/dt (h_ang/m)   = (1/m) · Σ (foot_i − CoM_world) × F_i
//! d/dt (base_pos)  = h_lin/m  −  ω × (R · com_offset_body)
//! d/dt (euler_ZYX) = T_zyx(euler) · ω
//! ```
//!
//! where `ω = R · I_body⁻¹ · R^T · h_ang` (rotated from body-frame
//! inertia to world-frame angular velocity), and `T_zyx` is the
//! kinematic transform from world-frame ω to ZYX-Euler derivatives.
//!
//! D1 scope is **just this dynamics function and its unit tests**.
//! D1.2 builds the discretised QP, D1.3 wires it into `WbcPipeline`.

use nalgebra::{Matrix3, Rotation3, Vector3};

/// 12-dim centroidal state. See module docs for the layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CentroidalState {
    /// CoM linear velocity in world frame (m/s).
    pub h_lin_per_mass: Vector3<f64>,
    /// Centroidal angular momentum / mass in world frame (m²/s).
    pub h_ang_per_mass: Vector3<f64>,
    /// Body root position in world frame (m).
    pub base_pos_world: Vector3<f64>,
    /// Base orientation as ZYX Euler angles (rad): `[roll, pitch, yaw]`.
    pub base_euler_zyx: Vector3<f64>,
}

impl CentroidalState {
    /// Pack into a flat 12-vector with layout `[h_lin/m; h_ang/m; pos; euler]`.
    pub fn to_vec12(&self) -> [f64; 12] {
        [
            self.h_lin_per_mass.x,
            self.h_lin_per_mass.y,
            self.h_lin_per_mass.z,
            self.h_ang_per_mass.x,
            self.h_ang_per_mass.y,
            self.h_ang_per_mass.z,
            self.base_pos_world.x,
            self.base_pos_world.y,
            self.base_pos_world.z,
            self.base_euler_zyx.x,
            self.base_euler_zyx.y,
            self.base_euler_zyx.z,
        ]
    }

    /// Inverse of [`Self::to_vec12`].
    pub fn from_vec12(v: &[f64; 12]) -> Self {
        Self {
            h_lin_per_mass: Vector3::new(v[0], v[1], v[2]),
            h_ang_per_mass: Vector3::new(v[3], v[4], v[5]),
            base_pos_world: Vector3::new(v[6], v[7], v[8]),
            base_euler_zyx: Vector3::new(v[9], v[10], v[11]),
        }
    }
}

/// 12-dim centroidal input: world-frame GRF per foot.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CentroidalInput {
    /// World-frame ground reaction force per foot, in canonical
    /// FL/FR/RL/RR slot order (N).
    pub grfs_world: [Vector3<f64>; 4],
}

impl CentroidalInput {
    /// Pack into a flat 12-vector.
    pub fn to_vec12(&self) -> [f64; 12] {
        [
            self.grfs_world[0].x, self.grfs_world[0].y, self.grfs_world[0].z,
            self.grfs_world[1].x, self.grfs_world[1].y, self.grfs_world[1].z,
            self.grfs_world[2].x, self.grfs_world[2].y, self.grfs_world[2].z,
            self.grfs_world[3].x, self.grfs_world[3].y, self.grfs_world[3].z,
        ]
    }

    pub fn from_vec12(v: &[f64; 12]) -> Self {
        Self {
            grfs_world: [
                Vector3::new(v[0], v[1], v[2]),
                Vector3::new(v[3], v[4], v[5]),
                Vector3::new(v[6], v[7], v[8]),
                Vector3::new(v[9], v[10], v[11]),
            ],
        }
    }
}

/// Constant model parameters for the type-1 centroidal SRBD MPC.
#[derive(Clone, Debug)]
pub struct CentroidalMpcConfig {
    /// Total robot mass (kg).
    pub mass_kg: f64,
    /// Centroidal angular inertia in body frame (kg·m²). Constant
    /// under type-1 SRBD — computed once at a nominal pose and held
    /// fixed across the horizon. The MPC user supplies this from
    /// `misarta::centroidal::compute_centroidal_inertia(model, q_nominal)`'s
    /// 3×3 angular block; an isotropic-inertia default is provided for
    /// quick-start tests.
    pub centroidal_inertia_body: Matrix3<f64>,
    /// CoM position relative to body root, expressed in body frame
    /// (m). For type-1 SRBD this is constant; the user supplies it
    /// from misarta's `compute_com` at the nominal pose. Zero default
    /// matches the textbook SRBD assumption.
    pub com_offset_body: Vector3<f64>,
    /// Friction coefficient for the pyramid constraint (dimensionless).
    pub friction_mu: f64,
    /// Max normal force per foot (N). 0 disables the upper bound.
    pub max_normal_force: f64,
    /// Prediction horizon length (number of discrete steps).
    pub horizon_steps: usize,
    /// Time per discrete step (s).
    pub dt_per_step: f64,
    /// State cost weights `Q_diag` (size 12), layout matches `to_vec12`.
    pub q_diag: [f64; 12],
    /// Input cost weight (scalar applied uniformly across all 12 GRF
    /// components per step).
    pub r_diag: f64,
}

impl Default for CentroidalMpcConfig {
    fn default() -> Self {
        // Cheetah-3 baseline: 9 kg body, isotropic ~0.2 kg·m² inertia.
        // The host should call `auto_detect_*` from articara::gait to
        // populate the right values for the specific URDF. These
        // defaults exist so unit tests can build a config without
        // pulling in an articara model.
        Self {
            mass_kg: 9.0,
            centroidal_inertia_body: Matrix3::from_diagonal(&Vector3::new(0.07, 0.26, 0.242)),
            com_offset_body: Vector3::zeros(),
            friction_mu: 0.5,
            max_normal_force: 200.0,
            horizon_steps: 10,
            dt_per_step: 0.030,
            q_diag: [
                // ḣ_lin/m (= v_com): light, the primary tracking variable
                1.0, 1.0, 1.0,
                // ḣ_ang/m: light
                0.5, 0.5, 5.0,
                // base_pos: bias toward the reference path, lateral / yaw heaviest
                0.0, 20.0, 50.0,
                // euler_zyx: keep level + track yaw
                25.0, 25.0, 50.0,
            ],
            r_diag: 1e-3,
        }
    }
}

/// Continuous-time centroidal-SRBD dynamics: ẋ = f(x, u).
///
/// Type-1 SRBD: centroidal inertia is treated as constant
/// (`cfg.centroidal_inertia_body`) — i.e. the legs' contribution to
/// the CMM is folded into a fixed nominal value rather than
/// re-evaluated at every node. This is what `legged_control` does
/// when `centroidalModelType = 1` and gives us the CoM-aware angular
/// dynamics without the per-node Pinocchio cost of the full
/// centroidal model.
///
/// Returned `CentroidalState` is the **time derivative** of the input
/// state (∈ ℝ¹² with the same field semantics as the input). A naïve
/// explicit-Euler integrator is `x_{k+1} = x_k + dt · f(x_k, u_k)`,
/// which is what the MPC uses for its discretised shooting model in
/// D1.2.
///
/// `foot_world` carries the per-leg world-frame foot position; in
/// the type-1 SRBD it's an external input (the QP's `r_per_leg`), so
/// callers compute it from FK + footstep planning *outside* this fn.
pub fn centroidal_dynamics(
    state: &CentroidalState,
    input: &CentroidalInput,
    foot_world: &[Vector3<f64>; 4],
    cfg: &CentroidalMpcConfig,
) -> CentroidalState {
    let g_world = Vector3::new(0.0, 0.0, -9.81);

    // Body-frame → world rotation from the state's Euler angles.
    let r_world_body = Rotation3::from_euler_angles(
        state.base_euler_zyx.x,
        state.base_euler_zyx.y,
        state.base_euler_zyx.z,
    );

    // CoM position in world frame: base root + R · com_offset.
    let com_offset_world = r_world_body * cfg.com_offset_body;
    let com_pos_world = state.base_pos_world + com_offset_world;

    // ── Linear momentum rate ────────────────────────────────────────
    //   d/dt (h_lin/m) = (Σ F)/m + g
    let total_f: Vector3<f64> = input.grfs_world.iter().sum();
    let h_lin_dot_per_m = total_f / cfg.mass_kg.max(1e-9) + g_world;

    // ── Angular momentum rate ───────────────────────────────────────
    //   d/dt (h_ang/m) = (1/m) · Σ (foot_i − CoM_world) × F_i
    let mut tau_world = Vector3::zeros();
    for slot in 0..4 {
        let r = foot_world[slot] - com_pos_world;
        tau_world += r.cross(&input.grfs_world[slot]);
    }
    let h_ang_dot_per_m = tau_world / cfg.mass_kg.max(1e-9);

    // ── Angular velocity in world frame ─────────────────────────────
    // ω_world = R · I_body⁻¹ · R^T · h_ang_world
    // (rotate h_ang into body frame, divide by body inertia, rotate
    // result back to world frame.)
    let h_ang_world = cfg.mass_kg * state.h_ang_per_mass;
    let i_body_inv = cfg
        .centroidal_inertia_body
        .try_inverse()
        .unwrap_or_else(Matrix3::identity);
    let r_mat = r_world_body.matrix();
    let h_ang_body = r_mat.transpose() * h_ang_world;
    let omega_body = i_body_inv * h_ang_body;
    let omega_world = r_mat * omega_body;

    // ── Base-position rate ──────────────────────────────────────────
    //   v_base_world = v_com_world − ω × com_offset_world
    // For zero offset this collapses to v_base = v_com = h_lin/m.
    let base_pos_dot = state.h_lin_per_mass - omega_world.cross(&com_offset_world);

    // ── Euler-ZYX rate from world-frame ω ───────────────────────────
    let base_euler_dot =
        euler_zyx_dot_from_world_omega(&state.base_euler_zyx, &omega_world);

    CentroidalState {
        h_lin_per_mass: h_lin_dot_per_m,
        h_ang_per_mass: h_ang_dot_per_m,
        base_pos_world: base_pos_dot,
        base_euler_zyx: base_euler_dot,
    }
}

/// Convert a world-frame angular velocity into ZYX-Euler-angle
/// derivatives. Convention: `R = R_z(yaw) · R_y(pitch) · R_x(roll)`,
/// `euler = [roll, pitch, yaw]`.
///
/// Derivation. The body-frame angular velocity satisfies the
/// Euler-rate kinematic equation `ω_body = T_body(euler) · euler_dot`
/// where for ZYX:
///
/// ```text
/// T_body = [ 1,    0,           -sin(pitch)         ]
///          [ 0,  cos(roll),  cos(pitch)·sin(roll)   ]
///          [ 0, -sin(roll),  cos(pitch)·cos(roll)   ]
/// ```
///
/// Inverting and pre-multiplying by `R^T` to convert world-frame ω
/// into body frame:
///
/// ```text
/// euler_dot = T_body⁻¹ · R^T · ω_world
/// ```
///
/// Equivalent to legged_control's `getEulerAnglesZyxDerivativesFromGlobalAngularVelocity`.
/// At identity orientation (euler = 0) this collapses to
/// `euler_dot = ω_world`, which is the test's sanity check.
///
/// Singular at `pitch = ±π/2` (gimbal lock). The controller's
/// operating regime keeps `|pitch| ≪ π/2`; the tiny `cp_safe` clamp
/// only fires for pathological inputs and bounds the divergence.
fn euler_zyx_dot_from_world_omega(
    euler: &Vector3<f64>,
    omega_world: &Vector3<f64>,
) -> Vector3<f64> {
    // Step 1: rotate ω into body frame. R = R_z(yaw)·R_y(pitch)·R_x(roll).
    let r_world_body = Rotation3::from_euler_angles(euler.x, euler.y, euler.z);
    let omega_body = r_world_body.transpose() * omega_world;

    // Step 2: apply T_body⁻¹ (standard ZYX body-frame Euler-rate inverse).
    let (sr, cr) = euler.x.sin_cos(); // roll
    let (sp, cp) = euler.y.sin_cos(); // pitch
    let cp_safe = if cp.abs() < 1e-6 { 1e-6_f64.copysign(cp) } else { cp };
    let tan_p = sp / cp_safe;

    let roll_dot = omega_body.x + sr * tan_p * omega_body.y + cr * tan_p * omega_body.z;
    let pitch_dot = cr * omega_body.y - sr * omega_body.z;
    let yaw_dot = sr / cp_safe * omega_body.y + cr / cp_safe * omega_body.z;

    Vector3::new(roll_dot, pitch_dot, yaw_dot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn nominal_namiashi_cfg() -> CentroidalMpcConfig {
        // namiashi-class quadruped: 2.4 kg, trunk-dominated inertia.
        CentroidalMpcConfig {
            mass_kg: 2.4,
            centroidal_inertia_body: Matrix3::from_diagonal(&Vector3::new(
                0.0019, 0.0086, 0.0090,
            )),
            com_offset_body: Vector3::zeros(),
            friction_mu: 0.5,
            max_normal_force: 100.0,
            horizon_steps: 10,
            dt_per_step: 0.030,
            q_diag: [
                1.0, 1.0, 1.0, 0.5, 0.5, 5.0, 0.0, 20.0, 50.0, 25.0, 25.0, 50.0,
            ],
            r_diag: 1e-3,
        }
    }

    /// Nominal stance footprint for a namiashi-class robot: 4 feet
    /// at world `(±0.147, ±0.109, 0)`.
    fn nominal_foot_world() -> [Vector3<f64>; 4] {
        [
            Vector3::new(0.147, 0.109, 0.0),   // FL
            Vector3::new(0.147, -0.109, 0.0),  // FR
            Vector3::new(-0.147, 0.109, 0.0),  // RL
            Vector3::new(-0.147, -0.109, 0.0), // RR
        ]
    }

    #[test]
    fn static_stand_yields_zero_state_derivative() {
        // Body upright at origin, no velocity, GRFs balance gravity.
        let cfg = nominal_namiashi_cfg();
        let state = CentroidalState {
            h_lin_per_mass: Vector3::zeros(),
            h_ang_per_mass: Vector3::zeros(),
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            base_euler_zyx: Vector3::zeros(),
        };
        let f_per_foot_z = cfg.mass_kg * 9.81 / 4.0;
        let input = CentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot_z); 4],
        };

        let dx = centroidal_dynamics(&state, &input, &nominal_foot_world(), &cfg);

        // ḣ_lin/m = 0 (Σ F = m·g·ẑ ⇒ Σf/m = (0, 0, g) and (0, 0, g) + g_world = 0).
        assert_abs_diff_eq!(dx.h_lin_per_mass.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_lin_per_mass.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_lin_per_mass.z, 0.0, epsilon = 1e-9);

        // ḣ_ang/m = 0 (symmetric stance, equal vertical force per foot).
        assert_abs_diff_eq!(dx.h_ang_per_mass.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_ang_per_mass.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_ang_per_mass.z, 0.0, epsilon = 1e-12);

        // Base position / orientation: zero v_com → zero base velocity.
        assert_abs_diff_eq!(dx.base_pos_world.norm(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.base_euler_zyx.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn free_fall_accelerates_at_g_downward() {
        // No contact forces → only gravity acts on CoM. Expected:
        //   ḣ_lin/m = (0, 0, -9.81), everything else zero.
        let cfg = nominal_namiashi_cfg();
        let state = CentroidalState::default();
        let input = CentroidalInput::default();

        let dx = centroidal_dynamics(&state, &input, &nominal_foot_world(), &cfg);

        assert_abs_diff_eq!(dx.h_lin_per_mass.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_lin_per_mass.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_lin_per_mass.z, -9.81, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_ang_per_mass.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn forward_force_on_front_legs_drives_x_acceleration() {
        // Front-leg pair (FL+FR) pushing forward at +x, no vertical
        // forces. Total Σ F = (2·Fx, 0, 0). Expect:
        //   ḣ_lin/m = (2·Fx/m, 0, -g)
        let cfg = nominal_namiashi_cfg();
        let state = CentroidalState::default();
        let fx = 1.0_f64;
        let input = CentroidalInput {
            grfs_world: [
                Vector3::new(fx, 0.0, 0.0),  // FL
                Vector3::new(fx, 0.0, 0.0),  // FR
                Vector3::zeros(),             // RL
                Vector3::zeros(),             // RR
            ],
        };

        let dx = centroidal_dynamics(&state, &input, &nominal_foot_world(), &cfg);

        assert_abs_diff_eq!(
            dx.h_lin_per_mass.x,
            2.0 * fx / cfg.mass_kg,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(dx.h_lin_per_mass.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.h_lin_per_mass.z, -9.81, epsilon = 1e-12);
    }

    #[test]
    fn asymmetric_diagonal_force_produces_yaw_moment() {
        // FL pushes +x, RR pushes -x. The two forces have equal
        // magnitude but opposite direction at diagonal feet. About
        // the CoM (= origin here, com_offset = 0) the moment arms are
        //   FL: ( 0.147,  0.109, 0)
        //   RR: (-0.147, -0.109, 0)
        // crosses with F = (±fx, 0, 0):
        //   r_FL × F_FL = (0.147, 0.109, 0) × (fx, 0, 0)
        //               = (0.109·0 − 0·0, 0·fx − 0.147·0, 0.147·0 − 0.109·fx)
        //               = (0, 0, -0.109·fx)
        //   r_RR × F_RR = (-0.147, -0.109, 0) × (-fx, 0, 0)
        //               = (0, 0, -(-0.109)·(-fx)) = (0, 0, -0.109·fx)
        // (signs intuitively: FL pushed forward + RR pushed backward
        // = couple yawing the body **clockwise** about +z, i.e.
        // negative yaw rate)
        // total τ_z = -0.218·fx. Then ḣ_ang/m.z = τ_z / m.
        let cfg = nominal_namiashi_cfg();
        let state = CentroidalState::default();
        let fx = 1.0_f64;
        let input = CentroidalInput {
            grfs_world: [
                Vector3::new(fx, 0.0, 0.0),
                Vector3::zeros(),
                Vector3::zeros(),
                Vector3::new(-fx, 0.0, 0.0),
            ],
        };

        let dx = centroidal_dynamics(&state, &input, &nominal_foot_world(), &cfg);

        assert_abs_diff_eq!(
            dx.h_ang_per_mass.z,
            -0.218 * fx / cfg.mass_kg,
            epsilon = 1e-9
        );
        // x and y angular components should be zero (forces are pure +x
        // and the moment arms have z = 0).
        assert_abs_diff_eq!(dx.h_ang_per_mass.x, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(dx.h_ang_per_mass.y, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn com_offset_creates_rolling_moment_under_gravity() {
        // The defining test for why this module exists. A body whose
        // CoM is offset +y from the body root, under gravity supported
        // by 4 symmetric feet, sees a rolling moment about +x:
        //   τ_x = -m·g·d_y_com_offset
        // (CoM is left of geometric centre → gravity rolls the body
        // to the left, body rotates about +x in world frame.)
        let mut cfg = nominal_namiashi_cfg();
        cfg.com_offset_body = Vector3::new(0.0, 0.005, 0.0); // +5 mm in body+y
        let state = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        // Symmetric vertical GRFs, total = m·g (so linear ḣ ≈ 0).
        let f_per_foot_z = cfg.mass_kg * 9.81 / 4.0;
        let input = CentroidalInput {
            grfs_world: [Vector3::new(0.0, 0.0, f_per_foot_z); 4],
        };

        let dx = centroidal_dynamics(&state, &input, &nominal_foot_world(), &cfg);

        // Linear momentum balanced.
        assert_abs_diff_eq!(dx.h_lin_per_mass.norm(), 0.0, epsilon = 1e-9);

        // Roll moment from CoM offset: each foot's GRF crosses with
        // (foot − CoM_world). CoM_world = base_pos + (0, 0.005, 0)
        // = (0, 0.005, 0.165). Sum (foot_i − CoM) × F_i:
        // FL: (0.147, 0.104, -0.165) × (0, 0, f_z) = (0.104·f_z, -0.147·f_z, 0)
        // FR: (0.147, -0.114, -0.165) × (0, 0, f_z) = (-0.114·f_z, -0.147·f_z, 0)
        // RL: (-0.147, 0.104, -0.165) × (0, 0, f_z) = (0.104·f_z, 0.147·f_z, 0)
        // RR: (-0.147, -0.114, -0.165) × (0, 0, f_z) = (-0.114·f_z, 0.147·f_z, 0)
        // Sum.x = (0.104 - 0.114 + 0.104 - 0.114) · f_z = -0.020 · f_z
        //       = -m·g/4 · 0.020 = -m·g · 0.005     ← exactly m·g·d_com_offset
        // Sum.y = (-0.147 - 0.147 + 0.147 + 0.147) · f_z = 0
        // Sum.z = 0
        let m = cfg.mass_kg;
        let g = 9.81;
        let d = 0.005;
        // ḣ_ang/m = τ_world / m = (-m·g·d, 0, 0) / m = (-g·d, 0, 0)
        assert_abs_diff_eq!(dx.h_ang_per_mass.x, -g * d, epsilon = 1e-9);
        assert_abs_diff_eq!(dx.h_ang_per_mass.y, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(dx.h_ang_per_mass.z, 0.0, epsilon = 1e-9);

        // The whole point — this is the rolling moment the body-root
        // SRBD MPC was missing.
        let _ = m;
    }

    #[test]
    fn euler_dot_at_zero_orientation_equals_world_omega() {
        // Identity orientation (euler = 0), world ω = body ω.
        // Then euler_dot = world_omega.
        let euler = Vector3::zeros();
        let omega = Vector3::new(0.1, 0.2, 0.3);
        let euler_dot = euler_zyx_dot_from_world_omega(&euler, &omega);
        assert_abs_diff_eq!(euler_dot.x, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(euler_dot.y, 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(euler_dot.z, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn state_pack_unpack_roundtrip() {
        let s = CentroidalState {
            h_lin_per_mass: Vector3::new(0.1, 0.2, 0.3),
            h_ang_per_mass: Vector3::new(0.4, 0.5, 0.6),
            base_pos_world: Vector3::new(1.0, 2.0, 3.0),
            base_euler_zyx: Vector3::new(0.01, 0.02, 0.03),
        };
        let v = s.to_vec12();
        let s2 = CentroidalState::from_vec12(&v);
        assert_eq!(s, s2);
    }

    #[test]
    fn input_pack_unpack_roundtrip() {
        let u = CentroidalInput {
            grfs_world: [
                Vector3::new(1.0, 2.0, 3.0),
                Vector3::new(4.0, 5.0, 6.0),
                Vector3::new(7.0, 8.0, 9.0),
                Vector3::new(10.0, 11.0, 12.0),
            ],
        };
        let v = u.to_vec12();
        let u2 = CentroidalInput::from_vec12(&v);
        assert_eq!(u, u2);
    }
}
