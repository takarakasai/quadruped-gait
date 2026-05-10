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

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

/// 12-dim centroidal state.
///
/// **State design (D1.4)**: stores `angular_velocity_world` directly
/// instead of the centroidal `h_ang/m`. The two are related by
/// `h_ang/m = (I_centroidal/m) · ω`, but their numerical scales
/// differ by `m/I` ≈ 267 for namiashi-class robots, which forces
/// 5-6 orders-of-magnitude weight ratios in the QP cost matrix and
/// breaks clarabel's tracking accuracy. Storing ω directly lets us
/// reuse SRBD's already-tuned cost weights without unit gymnastics.
///
/// The CoM-aware moment-arm structure (`α = I_com⁻¹ · Σ (foot_i −
/// CoM_world) × F_i`) is preserved in the dynamics — that's the
/// actual benefit of the centroidal model over the body-root SRBD.
/// Storing ω vs h_ang/m is a presentation choice that doesn't affect
/// the underlying physics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CentroidalState {
    /// CoM linear velocity in world frame (m/s). Equals "linear
    /// momentum / mass"; we keep the field name for backward
    /// compatibility with D1.1-D1.3 callers.
    pub h_lin_per_mass: Vector3<f64>,
    /// Body angular velocity in world frame (rad/s). For a rigid
    /// body the CoM and base share ω, so this works at either
    /// reference point. Same units as SRBD's body-frame ω so cost
    /// weights port directly.
    pub angular_velocity_world: Vector3<f64>,
    /// Body root position in world frame (m).
    pub base_pos_world: Vector3<f64>,
    /// Base orientation as ZYX Euler angles (rad): `[roll, pitch, yaw]`.
    pub base_euler_zyx: Vector3<f64>,
}

impl CentroidalState {
    /// Pack into a flat 12-vector with layout `[v_com; ω_world; pos; euler]`.
    pub fn to_vec12(&self) -> [f64; 12] {
        [
            self.h_lin_per_mass.x,
            self.h_lin_per_mass.y,
            self.h_lin_per_mass.z,
            self.angular_velocity_world.x,
            self.angular_velocity_world.y,
            self.angular_velocity_world.z,
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
            angular_velocity_world: Vector3::new(v[3], v[4], v[5]),
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
    /// SQP-style re-linearisation iterations within a single
    /// [`CentroidalMpc::solve`] call.
    ///
    /// The convex centroidal QP linearises angular dynamics around a
    /// reference yaw (`psi_ref`) and small roll/pitch. When the
    /// reference's predicted yaw evolves significantly across the
    /// horizon (e.g. yaw command at 0.5 rad/s × 300 ms ≈ 0.15 rad
    /// of yaw change) the linearisation error accumulates and the
    /// QP's optimum diverges from the true non-linear optimum.
    ///
    /// SQP fixes this by:
    ///   1. Solving the QP linearised at the reference (or previous
    ///      iteration's predicted trajectory).
    ///   2. Using the predicted trajectory from that QP as the new
    ///      linearisation point and re-solving.
    ///   3. Repeating until the trajectory converges or `sqp_iterations`
    ///      iterations have run.
    ///
    /// `1` (default) ⇒ single-shot solve (D1 behaviour, fastest).
    /// `2-3`         ⇒ typical SQP iteration count for legged_control-
    ///                 class controllers — improves angular tracking
    ///                 at non-zero yaw cmds at the cost of solver time.
    /// `>5`          ⇒ usually overkill; convergence stalls fast for
    ///                 short-horizon convex QPs.
    pub sqp_iterations: usize,
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
                // v_com (m/s): same scale as SRBD's `v`, weight 1.0.
                1.0, 1.0, 1.0,
                // ω_world (rad/s): identical to SRBD's `ω` weights.
                // D1.4 update — state[3..6] is now angular velocity
                // directly (was h_ang/m in D1.1-D1.3, with units that
                // forced 5+ orders of magnitude in the cost matrix).
                0.5, 0.5, 10.0,
                // base_pos (m): D1.4 reduced lateral / longitudinal
                // weights to 5 (from SRBD's 20) — centroidal QP's
                // I_world⁻¹·skew(r) entries are O(100) for namiashi
                // (vs SRBD's m·1/I ~ idem), but combined with the
                // CoM-shifted moment arm and explicit reference yaw,
                // 20 over-corrects and produces oscillating GRFs that
                // saturate the friction cone. 5 keeps tracking firm
                // without saturation.
                0.0, 5.0, 50.0,
                // euler_zyx (rad): same as SRBD `θ` weights —
                // keep body level + track yaw.
                25.0, 25.0, 50.0,
            ],
            r_diag: 1e-3,
            sqp_iterations: 1,
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

    // ── Angular acceleration ────────────────────────────────────────
    // Newton-Euler at CoM: α_world = I_world⁻¹ · (τ_world − ω × I·ω)
    //   I_world = R · I_body · R^T  (yaw-only rotation of body inertia)
    //   The Coriolis term `ω × (I·ω)` is `O(ω²)` and small at typical
    //   gait yaw rates; we keep it for accuracy at higher rates.
    let r_mat = r_world_body.matrix();
    let i_world = r_mat * cfg.centroidal_inertia_body * r_mat.transpose();
    let i_world_inv = i_world.try_inverse().unwrap_or_else(Matrix3::identity);
    let omega_world = state.angular_velocity_world;
    let i_omega = i_world * omega_world;
    let coriolis = omega_world.cross(&i_omega);
    let alpha_world = i_world_inv * (tau_world - coriolis);

    // ── Base-position rate ──────────────────────────────────────────
    //   v_base_world = v_com_world − ω × com_offset_world
    // For zero offset this collapses to v_base = v_com = h_lin/m.
    let base_pos_dot = state.h_lin_per_mass - omega_world.cross(&com_offset_world);

    // ── Euler-ZYX rate from world-frame ω ───────────────────────────
    let base_euler_dot =
        euler_zyx_dot_from_world_omega(&state.base_euler_zyx, &omega_world);

    CentroidalState {
        h_lin_per_mass: h_lin_dot_per_m,
        angular_velocity_world: alpha_world,
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

/// Predicted base acceleration (body root, world frame) given the
/// centroidal-SRBD model and the current MPC GRFs. Sibling of
/// [`crate::predicted_base_accel_world`] but uses the CoM-aware
/// moment arm `r_i = foot_world − CoM_world`, where `CoM_world =
/// body_root_world + R_world_body · cfg.com_offset_body`.
///
/// Used by the WBC pipeline to compute its `a_base_des` reference
/// when the host's gait controller is in `GaitMode::CentroidalSrbd`.
/// In that mode the MPC's predicted GRFs satisfy CoM Newton-Euler
/// (not body-root), so feeding them through the body-root
/// `predicted_base_accel_world` would create a 2-3% moment-arm
/// mismatch that the WBC then chases unsuccessfully — exactly the
/// failure mode that broke the C1 attempt.
///
/// Returns `(a_lin_world, a_ang_world)`. The linear accel is the same
/// as the body-root version (`Σf/m + g` is independent of where the
/// reference point sits — the kinematic difference between root and
/// CoM linear acceleration is `O(α · com_offset)`, well below WBC's
/// noise floor for typical 5 mm offsets and 1 rad/s² angular accels).
/// The angular accel uses the centroidal moment-arm correction.
pub fn predicted_base_accel_world_centroidal(
    cfg: &CentroidalMpcConfig,
    body_pos_world: Vector3<f64>,
    body_quat_to_world: nalgebra::UnitQuaternion<f64>,
    omega_obs_world: Vector3<f64>,
    grfs_world: &[Vector3<f64>; 4],
    foot_positions_world: &[Vector3<f64>; 4],
) -> (Vector3<f64>, Vector3<f64>) {
    let g_world = Vector3::new(0.0, 0.0, -9.81);

    // Linear: Σf/m + g.
    let total_f: Vector3<f64> = grfs_world.iter().sum();
    let a_lin_world = total_f / cfg.mass_kg.max(1e-9) + g_world;

    // CoM position in world.
    let r_mat = body_quat_to_world.to_rotation_matrix().into_inner();
    let com_offset_world = r_mat * cfg.com_offset_body;
    let com_pos_world = body_pos_world + com_offset_world;

    // Angular: α = I_world⁻¹ · (Σ (foot − CoM) × F − ω × Iω)
    let i_world = r_mat * cfg.centroidal_inertia_body * r_mat.transpose();
    let i_world_inv = i_world.try_inverse().unwrap_or_else(Matrix3::identity);
    let mut tau_world = Vector3::zeros();
    for slot in 0..4 {
        let r = foot_positions_world[slot] - com_pos_world;
        tau_world += r.cross(&grfs_world[slot]);
    }
    let i_omega = i_world * omega_obs_world;
    let coriolis = omega_obs_world.cross(&i_omega);
    let a_ang_world = i_world_inv * (tau_world - coriolis);

    (a_lin_world, a_ang_world)
}

/// Reference trajectory the centroidal MPC tracks. One state per
/// horizon step (length must equal `cfg.horizon_steps`).
#[derive(Clone, Debug)]
pub struct CentroidalReference {
    pub states: Vec<CentroidalState>,
}

impl CentroidalReference {
    /// Static reference: constant `s` over the horizon. Used for hover
    /// or hold-position tests.
    pub fn constant(s: CentroidalState, horizon_steps: usize) -> Self {
        Self {
            states: vec![s; horizon_steps],
        }
    }

    /// Constant-velocity reference: integrate base position + yaw
    /// forward from `s_now` at world-frame velocity `(v_world, wz)`.
    /// The host typically rotates a body-frame command into world
    /// frame using current yaw before calling this.
    pub fn from_constant_velocity(
        s_now: CentroidalState,
        v_world: Vector3<f64>,
        wz: f64,
        cfg: &CentroidalMpcConfig,
    ) -> Self {
        let mut states = Vec::with_capacity(cfg.horizon_steps);
        // Build the reference by stepping forward from now: position
        // and yaw integrate at the commanded velocity, momentum
        // tracks the commanded velocity exactly (so the MPC has a
        // reachable target rather than a step jump).
        let mut s = s_now;
        s.h_lin_per_mass = v_world;
        // Reference angular velocity in world frame: pure yaw at the
        // commanded rate (roll/pitch held at zero).
        s.angular_velocity_world = Vector3::new(0.0, 0.0, wz);
        // Integrate position + yaw across the horizon.
        for k in 0..cfg.horizon_steps {
            let t = (k + 1) as f64 * cfg.dt_per_step;
            let mut sk = s;
            sk.base_pos_world = s_now.base_pos_world + v_world * t;
            sk.base_euler_zyx.z = s_now.base_euler_zyx.z + wz * t;
            states.push(sk);
        }
        Self { states }
    }
}

/// Per-leg per-horizon-step world-frame vector **from the CoM to the
/// foot**. For type-1 SRBD where the CoM is constant in body frame,
/// the host computes this once per tick from FK + footstep planning
/// and passes it to [`CentroidalMpc::solve`].
#[derive(Clone, Debug)]
pub struct CentroidalFootOffsets {
    /// `r[leg][k]` = `foot_world(k) − CoM_world(k)`. Length per leg
    /// must equal `cfg.horizon_steps`.
    pub r: [Vec<Vector3<f64>>; 4],
}

impl CentroidalFootOffsets {
    /// All four legs share a fixed `r_per_leg` across the horizon.
    /// Useful for short-horizon MPC where the CoM doesn't move much.
    pub fn constant_per_leg(per_leg: [Vector3<f64>; 4], horizon_steps: usize) -> Self {
        Self {
            r: [
                vec![per_leg[0]; horizon_steps],
                vec![per_leg[1]; horizon_steps],
                vec![per_leg[2]; horizon_steps],
                vec![per_leg[3]; horizon_steps],
            ],
        }
    }
}

/// Per-leg per-step boolean stance schedule. `is_stance[leg][k] == true`
/// means foot `leg` may apply ground reaction force at step `k`; swing
/// legs have their GRF pinned to zero by an equality constraint.
#[derive(Clone, Debug)]
pub struct CentroidalContactSchedule {
    pub is_stance: [Vec<bool>; 4],
}

impl CentroidalContactSchedule {
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

/// Output of [`CentroidalMpc::solve`].
#[derive(Clone, Debug)]
pub struct CentroidalMpcSolution {
    /// First-step GRFs (world frame, N) — what the host commits this
    /// MPC tick. Receding-horizon: subsequent steps are diagnostic.
    pub grfs_first_step: [Vector3<f64>; 4],
    /// Full-horizon GRFs (one per step, per leg).
    pub grfs_all_steps: Vec<[Vector3<f64>; 4]>,
    /// Predicted state across the horizon from the QP-optimal U.
    pub predicted_body_states: Vec<CentroidalState>,
    /// QP objective value.
    pub objective: f64,
    /// Whether clarabel reported `Solved` or `AlmostSolved`. False
    /// indicates the host should fall back to a held GRF.
    pub solved: bool,
}

/// Stateless centroidal-SRBD MPC solver. Holds only the cached
/// `CentroidalMpcConfig`; the host can rebuild the QP each tick
/// without re-allocating the tuning struct.
#[derive(Clone, Debug)]
pub struct CentroidalMpc {
    cfg: CentroidalMpcConfig,
}

impl CentroidalMpc {
    pub fn new(cfg: CentroidalMpcConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &CentroidalMpcConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: CentroidalMpcConfig) {
        self.cfg = cfg;
    }

    /// Solve the MPC QP for the next horizon and return the optimal
    /// per-leg GRFs at the **first** step (receding-horizon convention).
    ///
    /// State layout (13-dim with augmented gravity column):
    /// ```text
    /// x = [ h_lin/m (3) ; h_ang/m (3) ; base_pos (3) ; euler_zyx (3) ; g_const = −9.81 ]
    /// ```
    ///
    /// Linearisation: yaw-only (roll = pitch = 0 reference). Inputs
    /// `feet.r[leg][k]` give the world-frame moment arm (foot − CoM)
    /// at each horizon step; the caller is responsible for projecting
    /// them forward (typically constant for type-1 SRBD).
    pub fn solve(
        &self,
        state_now: CentroidalState,
        reference: &CentroidalReference,
        contact: &CentroidalContactSchedule,
        feet: &CentroidalFootOffsets,
    ) -> CentroidalMpcSolution {
        let n = self.cfg.horizon_steps;
        assert_eq!(reference.states.len(), n, "ref length mismatch");
        for leg in 0..4 {
            assert_eq!(contact.is_stance[leg].len(), n);
            assert_eq!(feet.r[leg].len(), n);
        }

        let nx = 13_usize;
        let nu = 12_usize;
        let n_iter = self.cfg.sqp_iterations.max(1);

        // ── SQP iteration loop ──────────────────────────────────────
        // Each iteration:
        //   1. Build QP linearised at `psi_ref_per_step`.
        //   2. Solve, get optimal U and predicted state trajectory.
        //   3. Update `psi_ref_per_step` from the predicted yaw at each
        //      step. The next iteration linearises angular dynamics
        //      around the new (closer to optimal) trajectory.
        // For `sqp_iterations = 1` the loop runs once with the
        // reference trajectory's yaw — identical to D1's single-shot
        // behaviour, no extra cost.
        let mut psi_ref_per_step: Vec<f64> = reference
            .states
            .iter()
            .map(|s| s.base_euler_zyx.z)
            .collect();
        let mut last_solution: Option<CentroidalMpcSolution> = None;
        for iter in 0..n_iter {
            let sol = self.solve_one_iter(
                state_now,
                reference,
                contact,
                feet,
                &psi_ref_per_step,
                nx,
                nu,
                n,
            );
            // Update linearisation point for the next iteration (if any)
            // from the predicted yaw trajectory of this iteration's
            // solution. Skip if solver failed — keep the previous psi_ref.
            if iter + 1 < n_iter && sol.solved {
                for k in 0..n {
                    psi_ref_per_step[k] = sol.predicted_body_states[k].base_euler_zyx.z;
                }
            }
            last_solution = Some(sol);
        }
        last_solution.expect("at least one SQP iteration ran")
    }

    /// Build + solve one SQP iteration's QP at the given per-step
    /// linearisation yaw. Used by [`Self::solve`]'s outer loop. The
    /// non-yaw inputs (`reference`, `contact`, `feet`) and the QP
    /// dimensions (`nx`, `nu`, `n`) are passed in to avoid recomputing.
    fn solve_one_iter(
        &self,
        state_now: CentroidalState,
        reference: &CentroidalReference,
        contact: &CentroidalContactSchedule,
        feet: &CentroidalFootOffsets,
        psi_ref_per_step: &[f64],
        nx: usize,
        nu: usize,
        n: usize,
    ) -> CentroidalMpcSolution {
        // ── Build per-step continuous-time A_c, B_c, then discretise ──

        let mut a_d_per_step: Vec<DMatrix<f64>> = Vec::with_capacity(n);
        let mut b_d_per_step: Vec<DMatrix<f64>> = Vec::with_capacity(n);
        for k in 0..n {
            let psi_ref = psi_ref_per_step[k];
            let r_per_leg = [feet.r[0][k], feet.r[1][k], feet.r[2][k], feet.r[3][k]];
            let stance = [
                contact.is_stance[0][k],
                contact.is_stance[1][k],
                contact.is_stance[2][k],
                contact.is_stance[3][k],
            ];
            let (a_c, b_c) = self.continuous_dynamics(psi_ref, &r_per_leg, &stance);
            // Forward Euler discretisation: x_{k+1} = (I + dt·A) x_k + dt·B u_k.
            let mut a_d = DMatrix::<f64>::identity(nx, nx);
            a_d += &a_c * self.cfg.dt_per_step;
            let b_d = b_c * self.cfg.dt_per_step;
            a_d_per_step.push(a_d);
            b_d_per_step.push(b_d);
        }

        // ── Lifted dynamics: X = A_x x_0 + B_u U  ───────────────────
        let mut a_x = DMatrix::<f64>::zeros(nx * n, nx);
        let mut b_u = DMatrix::<f64>::zeros(nx * n, nu * n);
        let mut prod = DMatrix::<f64>::identity(nx, nx);
        for k in 0..n {
            prod = &a_d_per_step[k] * &prod;
            a_x.view_mut((k * nx, 0), (nx, nx)).copy_from(&prod);
            // Row k of B_u: contribution from each input u_j (j ≤ k).
            //   B_u[k,j] = (A_k · A_{k-1} · … · A_{j+1}) · B_j.
            let mut tail = DMatrix::<f64>::identity(nx, nx);
            for j in (0..=k).rev() {
                let block = &tail * &b_d_per_step[j];
                b_u.view_mut((k * nx, j * nu), (nx, nu)).copy_from(&block);
                if j > 0 {
                    tail = &tail * &a_d_per_step[j];
                }
            }
        }

        // ── Cost: J = ‖X − X_ref‖²_Q + ‖U‖²_R ──────────────────────
        // P = 2 (B_u^T Q B_u + R), q = 2 B_u^T Q (A_x x_0 − X_ref).
        let mut q_block = DMatrix::<f64>::zeros(nx * n, nx * n);
        for k in 0..n {
            for i in 0..12 {
                // Augmented gravity column has zero weight (state[12]
                // = −9.81 is a deterministic constant, not tracked).
                q_block[(k * nx + i, k * nx + i)] = self.cfg.q_diag[i];
            }
        }
        let mut r_block = DMatrix::<f64>::zeros(nu * n, nu * n);
        for i in 0..(nu * n) {
            r_block[(i, i)] = self.cfg.r_diag;
        }

        let x_ref = {
            let mut v = DVector::<f64>::zeros(nx * n);
            for k in 0..n {
                let s = state_to_vec13(&reference.states[k]);
                v.rows_mut(k * nx, nx).copy_from(&s);
            }
            v
        };
        let x_now = state_to_vec13(&state_now);
        let drift = &a_x * &x_now - &x_ref;

        let p_dense = 2.0 * (b_u.transpose() * &q_block * &b_u + &r_block);
        let q_vec = 2.0 * (b_u.transpose() * &q_block * &drift);

        // ── Constraints ────────────────────────────────────────────
        let (a_csc, b_vec, cones) = build_constraints(&self.cfg, contact, n, nu);

        // ── clarabel solve ─────────────────────────────────────────
        let p_csc = dense_to_csc_upper(&p_dense);
        let q_slice: Vec<f64> = q_vec.iter().copied().collect();
        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.max_iter = 50;
        let mut solver =
            match DefaultSolver::new(&p_csc, &q_slice, &a_csc, &b_vec, &cones, settings) {
                Ok(s) => s,
                Err(_) => {
                    return CentroidalMpcSolution {
                        grfs_first_step: [Vector3::zeros(); 4],
                        grfs_all_steps: vec![[Vector3::zeros(); 4]; n],
                        predicted_body_states: vec![state_now; n],
                        objective: f64::NAN,
                        solved: false,
                    };
                }
            };
        solver.solve();

        let solved = matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        );
        let u_opt = &solver.solution.x;
        let objective = solver.solution.obj_val;

        // Decode U → per-leg per-step GRFs.
        let mut grfs_all_steps = Vec::with_capacity(n);
        for k in 0..n {
            let base = k * nu;
            grfs_all_steps.push([
                Vector3::new(u_opt[base], u_opt[base + 1], u_opt[base + 2]),
                Vector3::new(u_opt[base + 3], u_opt[base + 4], u_opt[base + 5]),
                Vector3::new(u_opt[base + 6], u_opt[base + 7], u_opt[base + 8]),
                Vector3::new(u_opt[base + 9], u_opt[base + 10], u_opt[base + 11]),
            ]);
        }

        // Decode predicted states from X = A_x x_0 + B_u U.
        let u_dvec = DVector::from_vec(u_opt.clone());
        let x_horizon = &a_x * &x_now + &b_u * &u_dvec;
        let mut predicted_body_states = Vec::with_capacity(n);
        for k in 0..n {
            let row0 = k * nx;
            predicted_body_states.push(CentroidalState {
                h_lin_per_mass: Vector3::new(
                    x_horizon[row0],
                    x_horizon[row0 + 1],
                    x_horizon[row0 + 2],
                ),
                angular_velocity_world: Vector3::new(
                    x_horizon[row0 + 3],
                    x_horizon[row0 + 4],
                    x_horizon[row0 + 5],
                ),
                base_pos_world: Vector3::new(
                    x_horizon[row0 + 6],
                    x_horizon[row0 + 7],
                    x_horizon[row0 + 8],
                ),
                base_euler_zyx: Vector3::new(
                    x_horizon[row0 + 9],
                    x_horizon[row0 + 10],
                    x_horizon[row0 + 11],
                ),
            });
        }

        CentroidalMpcSolution {
            grfs_first_step: grfs_all_steps[0],
            grfs_all_steps,
            predicted_body_states,
            objective,
            solved,
        }
    }

    /// Continuous-time A and B matrices for the discretised MPC at
    /// one horizon step. State layout (13 dim, augmented):
    ///
    /// ```text
    /// x = [ h_lin/m (0..3); h_ang/m (3..6); base_pos (6..9); euler (9..12); g_const (12) ]
    /// ```
    ///
    /// Yaw-only linearisation (roll = pitch = 0 in the reference).
    /// `r_per_leg[i]` = world-frame vector from CoM to foot i at this
    /// step.
    fn continuous_dynamics(
        &self,
        psi_ref: f64,
        r_per_leg: &[Vector3<f64>; 4],
        stance: &[bool; 4],
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let nx = 13;
        let nu = 12;
        let mut a = DMatrix::<f64>::zeros(nx, nx);
        let mut b = DMatrix::<f64>::zeros(nx, nu);

        let m = self.cfg.mass_kg.max(1e-9);

        // Yaw rotation R_z(ψ_ref) (small-angle on roll/pitch).
        let (s, c) = psi_ref.sin_cos();
        let r_z = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);

        // I_world = R_z · I_body · R_z^T  (yaw-only rotation).
        let i_body = self.cfg.centroidal_inertia_body;
        let i_world = r_z * i_body * r_z.transpose();
        let i_world_inv = i_world.try_inverse().unwrap_or_else(Matrix3::identity);

        // ── d/dt(h_lin/m) = (Σ F)/m + g ─────────────────────────────
        //   Linear in F: B[0..3, leg*3..(leg+1)*3] = (1/m)·I per stance leg.
        //   Gravity: state[12] = −9.81 holds the constant; A[2, 12] = 1
        //   so d(h_lin_z/m)/dt picks up state[12] = −9.81.
        a[(2, 12)] = 1.0;
        for leg in 0..4 {
            if !stance[leg] {
                continue;
            }
            for i in 0..3 {
                b[(i, leg * 3 + i)] = 1.0 / m;
            }
        }

        // ── d/dt(ω_world) = I_world⁻¹ · Σ (foot_i − CoM) × F_i ─────
        //   At ω≈0 reference the Coriolis term `ω × Iω` is dropped from
        //   the linearisation. The CoM offset enters via `r_per_leg`
        //   (= foot_world − CoM_world) — the controller subtracts the
        //   CoM offset from foot_world before passing to the QP.
        //   skew(r) is the 3×3 cross-product matrix; pre-multiplying
        //   by I_world⁻¹ gives the angular-acceleration response per
        //   foot force.
        for leg in 0..4 {
            if !stance[leg] {
                continue;
            }
            let r = r_per_leg[leg];
            let block = i_world_inv * skew_symmetric_3(&r);
            for i in 0..3 {
                for j in 0..3 {
                    b[(3 + i, leg * 3 + j)] = block[(i, j)];
                }
            }
        }

        // ── d/dt(base_pos) = v_com  (small-CoM-offset approximation) ─
        //   Strictly: base_pos_dot = v_com − ω × com_offset_world.
        //   At ω=0 reference and small com_offset (~5 mm) the cross-
        //   product term contributes O(α·offset) ≈ mm/s², below the
        //   QP's noise floor — drop it for the linearisation.
        a[(6, 0)] = 1.0;
        a[(7, 1)] = 1.0;
        a[(8, 2)] = 1.0;

        // ── d/dt(euler_zyx) = T_inv · ω_world ──────────────────────
        //   At euler_ref = (0, 0, ψ_ref): T_inv = R_z(ψ_ref)^T (because
        //   the body→world rotation at zero roll/pitch is just R_z).
        //   ω_world is now state field 3..6 directly (no I/m scaling).
        //   ⇒ A[9..12, 3..6] = R_z^T.
        let r_z_t = r_z.transpose();
        for i in 0..3 {
            for j in 0..3 {
                a[(9 + i, 3 + j)] = r_z_t[(i, j)];
            }
        }

        // Suppress the now-unused `m` if nobody else references it.
        let _ = i_world; // kept for symmetry; the linearisation only
                         // uses `i_world_inv` above.
        let _ = m;

        (a, b)
    }
}

fn state_to_vec13(s: &CentroidalState) -> DVector<f64> {
    let mut v = DVector::zeros(13);
    v[0] = s.h_lin_per_mass.x;
    v[1] = s.h_lin_per_mass.y;
    v[2] = s.h_lin_per_mass.z;
    v[3] = s.angular_velocity_world.x;
    v[4] = s.angular_velocity_world.y;
    v[5] = s.angular_velocity_world.z;
    v[6] = s.base_pos_world.x;
    v[7] = s.base_pos_world.y;
    v[8] = s.base_pos_world.z;
    v[9] = s.base_euler_zyx.x;
    v[10] = s.base_euler_zyx.y;
    v[11] = s.base_euler_zyx.z;
    v[12] = -9.81; // constant gravity column
    v
}

fn skew_symmetric_3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Convert a dense symmetric PSD matrix to clarabel's CSC upper-
/// triangular layout. (Same helper as `srbd_mpc.rs`; kept here so the
/// new module has no internal cross-module dependency on private
/// helpers.)
fn dense_to_csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.nrows();
    debug_assert_eq!(n, p.ncols());
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for i in 0..=j {
            let v = p[(i, j)];
            if v.abs() > 1e-12 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Build the friction-cone + contact-mode constraints. Layout matches
/// `srbd_mpc::build_constraints`: equality rows for swing legs first,
/// then inequality rows for stance legs (f_z bounds + 4-side pyramid).
fn build_constraints(
    cfg: &CentroidalMpcConfig,
    contact: &CentroidalContactSchedule,
    n: usize,
    nu: usize,
) -> (CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
    let total_vars = nu * n;
    let mu = cfg.friction_mu;
    let f_max = cfg.max_normal_force;

    let mut n_eq = 0;
    let mut n_ineq = 0;
    for k in 0..n {
        for leg in 0..4 {
            if contact.is_stance[leg][k] {
                let mut count = 4; // friction pyramid (±x, ±y vs μ·f_z)
                count += 1; // f_z ≥ 0
                if f_max > 0.0 {
                    count += 1; // f_z ≤ f_max
                }
                n_ineq += count;
            } else {
                n_eq += 3; // f_x = f_y = f_z = 0
            }
        }
    }

    let n_rows = n_eq + n_ineq;
    let mut a_dense = DMatrix::<f64>::zeros(n_rows, total_vars);
    let mut b_vec = vec![0.0; n_rows];
    let mut row = 0;

    // Equality rows
    for k in 0..n {
        for leg in 0..4 {
            if !contact.is_stance[leg][k] {
                let col = k * nu + leg * 3;
                for ax in 0..3 {
                    a_dense[(row + ax, col + ax)] = 1.0;
                }
                row += 3;
            }
        }
    }

    // Inequality rows
    for k in 0..n {
        for leg in 0..4 {
            if !contact.is_stance[leg][k] {
                continue;
            }
            let col_x = k * nu + leg * 3;
            let col_y = col_x + 1;
            let col_z = col_x + 2;
            // f_z ≥ 0  ⇒  -f_z ≤ 0
            a_dense[(row, col_z)] = -1.0;
            row += 1;
            // f_z ≤ f_max
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

    let a_csc = dense_to_csc(&a_dense);
    let cones = vec![
        SupportedConeT::ZeroConeT(n_eq),
        SupportedConeT::NonnegativeConeT(n_ineq),
    ];
    (a_csc, b_vec, cones)
}

/// CSC conversion of a general (not necessarily symmetric) dense matrix.
fn dense_to_csc(m: &DMatrix<f64>) -> CscMatrix<f64> {
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
                // v_com, ω_world, base_pos, euler — SRBD-equivalent weights.
                1.0, 1.0, 1.0, 0.5, 0.5, 10.0, 0.0, 20.0, 50.0, 25.0, 25.0, 50.0,
            ],
            r_diag: 1e-3,
            sqp_iterations: 1,
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
            angular_velocity_world: Vector3::zeros(),
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
        assert_abs_diff_eq!(dx.angular_velocity_world.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.angular_velocity_world.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx.angular_velocity_world.z, 0.0, epsilon = 1e-12);

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
        assert_abs_diff_eq!(dx.angular_velocity_world.norm(), 0.0, epsilon = 1e-12);
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
        // total τ_z = -0.218·fx. Then α_z = τ_z / I_z (D1.4: state[3..6]
        // is now ω directly so the time derivative is angular accel,
        // not angular momentum rate / mass).
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

        let i_z = cfg.centroidal_inertia_body[(2, 2)];
        assert_abs_diff_eq!(
            dx.angular_velocity_world.z,
            -0.218 * fx / i_z,
            epsilon = 1e-9
        );
        // x and y angular components should be zero (forces are pure +x
        // and the moment arms have z = 0).
        assert_abs_diff_eq!(dx.angular_velocity_world.x, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(dx.angular_velocity_world.y, 0.0, epsilon = 1e-9);
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
        // D1.4: state stores ω directly, so the time derivative is
        // angular accel α = I⁻¹·τ. With τ_x = -m·g·d:
        //   α_x = -m·g·d / I_xx
        let i_xx = cfg.centroidal_inertia_body[(0, 0)];
        let expected_alpha_x = -m * g * d / i_xx;
        assert_abs_diff_eq!(dx.angular_velocity_world.x, expected_alpha_x, epsilon = 1e-9);
        assert_abs_diff_eq!(dx.angular_velocity_world.y, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(dx.angular_velocity_world.z, 0.0, epsilon = 1e-9);
        // The whole point — this is the rolling moment the body-root
        // SRBD MPC was missing.
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
            angular_velocity_world: Vector3::new(0.4, 0.5, 0.6),
            base_pos_world: Vector3::new(1.0, 2.0, 3.0),
            base_euler_zyx: Vector3::new(0.01, 0.02, 0.03),
        };
        let v = s.to_vec12();
        let s2 = CentroidalState::from_vec12(&v);
        assert_eq!(s, s2);
    }

    /// Solver smoke test: hover scenario. All four feet in stance,
    /// reference is "hold position", state at rest. The QP should
    /// settle on Σ F_z ≈ m·g and zero horizontal forces.
    #[test]
    fn mpc_hover_balances_gravity() {
        let cfg = nominal_namiashi_cfg();
        let mpc = CentroidalMpc::new(cfg.clone());

        let s = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        let reference = CentroidalReference::constant(s, cfg.horizon_steps);
        let contact = CentroidalContactSchedule::all_stance(cfg.horizon_steps);
        // Feet at the nominal stance footprint, CoM at body root (no
        // offset in this cfg). r = foot_world - com_world = nominal_foot_world
        // since base_pos_world = 0 here.
        let nominal_feet = nominal_foot_world();
        let feet = CentroidalFootOffsets::constant_per_leg(nominal_feet, cfg.horizon_steps);

        let sol = mpc.solve(s, &reference, &contact, &feet);

        assert!(sol.solved, "MPC failed to solve hover scenario");
        let total_fz: f64 = sol.grfs_first_step.iter().map(|f| f.z).sum();
        let m_g = cfg.mass_kg * 9.81;
        // Allow 15% slack — clarabel + the small horizon's discretization
        // error means the equilibrium isn't reached to machine precision
        // at horizon=10.
        assert!(
            (total_fz - m_g).abs() < 0.15 * m_g,
            "hover: Σf_z = {:.3} N, expected ≈ m·g = {:.3} N",
            total_fz, m_g,
        );
        // Horizontal forces should be near zero.
        let total_fx: f64 = sol.grfs_first_step.iter().map(|f| f.x).sum();
        let total_fy: f64 = sol.grfs_first_step.iter().map(|f| f.y).sum();
        assert!(
            total_fx.abs() < 0.1 * m_g,
            "hover: Σf_x = {:.3} N, expected ≈ 0", total_fx
        );
        assert!(
            total_fy.abs() < 0.1 * m_g,
            "hover: Σf_y = {:.3} N, expected ≈ 0", total_fy
        );
    }

    /// Lateral velocity command (`vy = +0.10`) should produce a
    /// positive total Σ f_y (push CoM in +y, i.e. body-frame left).
    /// Mirror of `mpc_forward_cmd_yields_positive_fx` for the
    /// y-axis — catches sign / linearisation regressions in the
    /// lateral channel.
    #[test]
    fn mpc_lateral_cmd_yields_positive_fy() {
        let cfg = nominal_namiashi_cfg();
        let mpc = CentroidalMpc::new(cfg.clone());

        let s_now = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        let reference = CentroidalReference::from_constant_velocity(
            s_now,
            Vector3::new(0.0, 0.10, 0.0),
            0.0,
            &cfg,
        );
        let contact = CentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let feet = CentroidalFootOffsets::constant_per_leg(
            nominal_foot_world(),
            cfg.horizon_steps,
        );

        let sol = mpc.solve(s_now, &reference, &contact, &feet);
        assert!(sol.solved, "MPC failed to solve lateral-walk reference");

        let total_fy: f64 = sol.grfs_first_step.iter().map(|f| f.y).sum();
        assert!(
            total_fy > 0.05,
            "lateral cmd: Σf_y = {:.3} N, expected > 0.05 N",
            total_fy
        );
    }

    /// Compares `predicted_base_accel_world_centroidal` against the
    /// body-root SRBD `predicted_base_accel_world` at static stand
    /// with zero CoM offset. They MUST agree to numerical tolerance —
    /// any drift here means our centroidal formula has a sign /
    /// formula bug, not just a CoM-offset model difference.
    #[test]
    fn centroidal_and_srbd_accel_agree_at_zero_com_offset() {
        let mut cent_cfg = nominal_namiashi_cfg();
        cent_cfg.com_offset_body = Vector3::zeros();
        let srbd_cfg = crate::srbd_mpc::SrbdMpcConfig {
            mass_kg: cent_cfg.mass_kg,
            inertia_diag_body: Vector3::new(
                cent_cfg.centroidal_inertia_body[(0, 0)],
                cent_cfg.centroidal_inertia_body[(1, 1)],
                cent_cfg.centroidal_inertia_body[(2, 2)],
            ),
            ..crate::srbd_mpc::SrbdMpcConfig::default()
        };

        let body_pos = Vector3::new(0.0, 0.0, 0.165);
        let body_quat = nalgebra::UnitQuaternion::identity();
        let omega_world = Vector3::zeros();
        let v_obs_world = Vector3::zeros();
        // Asymmetric GRFs (so τ ≠ 0) — pure stand would give τ = 0
        // and trivially agree; this catches the moment-arm formula.
        let f = [
            Vector3::new(0.5, 0.3, 6.0),
            Vector3::new(-0.2, -0.1, 5.5),
            Vector3::new(0.1, 0.4, 6.5),
            Vector3::new(-0.3, -0.2, 5.0),
        ];
        let foot = nominal_foot_world();

        let (a_lin_cent, a_ang_cent) = predicted_base_accel_world_centroidal(
            &cent_cfg, body_pos, body_quat, omega_world, &f, &foot,
        );
        let srbd_state = crate::srbd_mpc::SrbdState {
            orientation_rpy: Vector3::zeros(),
            position: body_pos,
            // SRBD wants ω in body frame; for identity orientation
            // body == world.
            angular_velocity: omega_world,
            linear_velocity: v_obs_world,
        };
        let (a_lin_srbd, a_ang_srbd) =
            crate::srbd_mpc::predicted_base_accel_world(&srbd_cfg, &srbd_state, &f, &foot);

        for i in 0..3 {
            assert_abs_diff_eq!(
                a_lin_cent[i], a_lin_srbd[i],
                epsilon = 1e-9,
            );
            assert_abs_diff_eq!(
                a_ang_cent[i], a_ang_srbd[i],
                epsilon = 1e-9,
            );
        }
    }

    /// SQP iterations smoke test: at `sqp_iterations = 3`, solving a
    /// yaw-cmd reference produces a solution whose objective is no
    /// worse than the single-shot baseline (`sqp_iterations = 1`).
    /// Re-linearising at the predicted trajectory should never make
    /// the cost go up — that would mean the QP found a worse optimum
    /// than the reference linearisation, which is impossible for a
    /// convex problem barring numerical noise.
    #[test]
    fn mpc_sqp_iter_does_not_increase_objective() {
        let mut cfg_single = nominal_namiashi_cfg();
        cfg_single.sqp_iterations = 1;
        let mut cfg_iter = nominal_namiashi_cfg();
        cfg_iter.sqp_iterations = 3;

        let s_now = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        let reference = CentroidalReference::from_constant_velocity(
            s_now,
            Vector3::zeros(),
            0.5, // yaw rate
            &cfg_single,
        );
        let contact = CentroidalContactSchedule::all_stance(cfg_single.horizon_steps);
        let feet = CentroidalFootOffsets::constant_per_leg(
            nominal_foot_world(),
            cfg_single.horizon_steps,
        );

        let mpc_single = CentroidalMpc::new(cfg_single);
        let mpc_iter = CentroidalMpc::new(cfg_iter);
        let s = mpc_single.solve(s_now, &reference, &contact, &feet);
        let i = mpc_iter.solve(s_now, &reference, &contact, &feet);

        assert!(s.solved && i.solved);
        // The 3-iteration objective should be at most the 1-iteration
        // objective (within numerical tolerance).
        assert!(
            i.objective <= s.objective + 1e-6,
            "SQP 3-iter obj ({:.6}) > 1-iter obj ({:.6})",
            i.objective, s.objective,
        );
    }

    /// Yaw rate command (`wz = +0.5`) should produce a positive
    /// total moment about +z (body-frame yaw CCW). Verifies the
    /// angular dynamics linearisation tracks ω_z correctly.
    #[test]
    fn mpc_yaw_cmd_yields_positive_yaw_moment() {
        let cfg = nominal_namiashi_cfg();
        let mpc = CentroidalMpc::new(cfg.clone());

        let s_now = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        let reference = CentroidalReference::from_constant_velocity(
            s_now,
            Vector3::zeros(),
            0.5, // wz cmd
            &cfg,
        );
        let contact = CentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let feet = CentroidalFootOffsets::constant_per_leg(
            nominal_foot_world(),
            cfg.horizon_steps,
        );

        let sol = mpc.solve(s_now, &reference, &contact, &feet);
        assert!(sol.solved, "MPC failed to solve yaw-walk reference");

        // Total moment about +z from per-foot GRFs at nominal stance:
        //   τ_z = Σ (foot_i.x · F_i.y − foot_i.y · F_i.x)
        let feet_w = nominal_foot_world();
        let total_tau_z: f64 = sol
            .grfs_first_step
            .iter()
            .zip(feet_w.iter())
            .map(|(f, foot)| foot.x * f.y - foot.y * f.x)
            .sum();
        assert!(
            total_tau_z > 0.01,
            "yaw cmd: Σ τ_z = {:.3} N·m, expected > 0.01 N·m",
            total_tau_z
        );
    }

    /// Forward-velocity command produces a forward bias in the
    /// per-foot horizontal forces. Specifically, for `v_cmd = (+0.15, 0, 0)`
    /// the QP should choose a positive total Σ f_x (push the CoM in +x).
    #[test]
    fn mpc_forward_cmd_yields_positive_fx() {
        let cfg = nominal_namiashi_cfg();
        let mpc = CentroidalMpc::new(cfg.clone());

        let s_now = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        let reference = CentroidalReference::from_constant_velocity(
            s_now,
            Vector3::new(0.15, 0.0, 0.0),
            0.0,
            &cfg,
        );
        let contact = CentroidalContactSchedule::all_stance(cfg.horizon_steps);
        let feet = CentroidalFootOffsets::constant_per_leg(
            nominal_foot_world(),
            cfg.horizon_steps,
        );

        let sol = mpc.solve(s_now, &reference, &contact, &feet);
        assert!(sol.solved, "MPC failed to solve forward-walk reference");

        let total_fx: f64 = sol.grfs_first_step.iter().map(|f| f.x).sum();
        // The MPC should select a non-zero forward force to start
        // accelerating the CoM toward `v_cmd`. Sign matters: positive
        // x-force = body accelerates forward.
        assert!(
            total_fx > 0.05,
            "forward cmd: Σf_x = {:.3} N, expected > 0.05 N",
            total_fx
        );
    }

    /// Swing legs (no stance) have their GRF pinned to zero by the
    /// equality constraints.
    #[test]
    fn mpc_swing_leg_grf_is_zero() {
        let cfg = nominal_namiashi_cfg();
        let mpc = CentroidalMpc::new(cfg.clone());

        let s = CentroidalState {
            base_pos_world: Vector3::new(0.0, 0.0, 0.165),
            ..Default::default()
        };
        let reference = CentroidalReference::constant(s, cfg.horizon_steps);
        // Trot phase 1: FL and RR in stance, FR and RL in swing.
        let mut contact = CentroidalContactSchedule::all_stance(cfg.horizon_steps);
        contact.is_stance[1] = vec![false; cfg.horizon_steps]; // FR swing
        contact.is_stance[2] = vec![false; cfg.horizon_steps]; // RL swing
        let feet = CentroidalFootOffsets::constant_per_leg(
            nominal_foot_world(),
            cfg.horizon_steps,
        );

        let sol = mpc.solve(s, &reference, &contact, &feet);
        assert!(sol.solved);

        // FR and RL should have ≈ zero force (clarabel's equality
        // tolerance ~1e-5).
        let fr_norm = sol.grfs_first_step[1].norm();
        let rl_norm = sol.grfs_first_step[2].norm();
        assert!(fr_norm < 1e-4, "FR (swing) f = {:.6} N, expected 0", fr_norm);
        assert!(rl_norm < 1e-4, "RL (swing) f = {:.6} N, expected 0", rl_norm);

        // FL and RR (stance pair) must split the gravity load.
        let fl_fz = sol.grfs_first_step[0].z;
        let rr_fz = sol.grfs_first_step[3].z;
        let m_g = cfg.mass_kg * 9.81;
        assert!(
            (fl_fz + rr_fz - m_g).abs() < 0.2 * m_g,
            "diagonal stance Σf_z = {:.3} N, expected ≈ m·g = {:.3} N",
            fl_fz + rr_fz, m_g,
        );
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
