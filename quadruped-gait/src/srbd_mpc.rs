//! Convex MPC for ground reaction forces, after Di Carlo et al. (2018)
//! "Dynamic Locomotion in the MIT Cheetah 3 through Convex
//! Model-Predictive Control".
//!
//! # Model
//!
//! Single-rigid-body dynamics (SRBD) — the 13-dim state augments
//! position + orientation + linear/angular velocity with a constant
//! gravity term so the dynamics matrix `A` is time-invariant in
//! continuous time:
//!
//! ```text
//! x = [θ_x, θ_y, θ_z, p_x, p_y, p_z, ω_x, ω_y, ω_z, v_x, v_y, v_z, g]
//!     ↑──RPY──↑   ↑─world pos─↑  ↑──ang vel──↑   ↑──lin vel──↑   ↑gravity↑
//! ```
//!
//! Inputs: 12 GRF components (3 axes × 4 feet, world frame), one set
//! per discrete time step over the prediction horizon `N`.
//!
//! # QP formulation
//!
//! Condensed (substitute the dynamics so the QP variables are only the
//! inputs `U = [u_0; u_1; …; u_{N-1}]`):
//!
//! ```text
//! min  ½ U^T P U + q^T U
//! s.t. f_z ∈ [0, f_max]                             (per stance foot, per step)
//!      |f_x|, |f_y| ≤ μ · f_z                       (friction pyramid)
//!      f_i = 0                                       (per swing foot, per step)
//! ```
//!
//! `P = 2·(B_u^T Q B_u + R)` and `q = 2·B_u^T Q (A_x x_0 − X_ref)`,
//! where `A_x` and `B_u` are the lifted dynamics matrices that
//! reconstruct `X = [x_1; …; x_N]` from `x_0` and `U`.
//!
//! # What this is NOT
//!
//! - Not a torque-level controller. The output is **desired ground
//!   reaction forces**; converting them to joint torques requires a
//!   Jacobian-transpose mapping AND a torque-actuator chain in
//!   articara's `mujoco_sim` (Phase 4 work — currently the robot is
//!   driven by the position-control footstep planner in
//!   [`crate::MpcGaitController`] and these GRFs are visualisation /
//!   diagnostic only).

use clarabel::algebra::{CscMatrix, FloatT};
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus, ZeroConeT,
};
use nalgebra::{DMatrix, DVector, Vector3};

use crate::config::LegId;

/// 13-dim SRBD state.
#[derive(Clone, Copy, Debug, Default)]
pub struct SrbdState {
    pub orientation_rpy: Vector3<f64>,   // body roll, pitch, yaw
    pub position: Vector3<f64>,          // world-frame CoM
    pub angular_velocity: Vector3<f64>,  // body frame
    pub linear_velocity: Vector3<f64>,   // world frame
}

impl SrbdState {
    fn to_vec(self) -> DVector<f64> {
        let mut v = DVector::zeros(13);
        v[0] = self.orientation_rpy.x;
        v[1] = self.orientation_rpy.y;
        v[2] = self.orientation_rpy.z;
        v[3] = self.position.x;
        v[4] = self.position.y;
        v[5] = self.position.z;
        v[6] = self.angular_velocity.x;
        v[7] = self.angular_velocity.y;
        v[8] = self.angular_velocity.z;
        v[9] = self.linear_velocity.x;
        v[10] = self.linear_velocity.y;
        v[11] = self.linear_velocity.z;
        v[12] = -9.81; // constant gravity row, accel
        v
    }
}

/// MPC tuning weights and physical params. Defaults match the Di Carlo
/// 2018 paper for a Cheetah-class quadruped (~9 kg). Hosts running
/// heavier robots should crank up `mass_kg` and `inertia_diag_body`.
#[derive(Clone, Debug)]
pub struct SrbdMpcConfig {
    /// Prediction horizon length (number of discrete steps).
    pub horizon_steps: usize,
    /// Time per discrete step (s). `horizon_steps * dt_per_step` is
    /// the total prediction window — Di Carlo 2018 uses ~300 ms.
    pub dt_per_step: f64,
    /// Body mass (kg).
    pub mass_kg: f64,
    /// Diagonal of the body inertia in body frame (kg·m²).
    pub inertia_diag_body: Vector3<f64>,
    /// Friction coefficient for the pyramid constraint.
    pub friction_mu: f64,
    /// Max normal force per foot (N). 0 disables the upper bound.
    pub max_normal_force: f64,
    /// State cost weights `Q_diag` (size 13): `[θ; p; ω; v; g]` order.
    pub q_diag: [f64; 13],
    /// Input cost weight (scalar applied uniformly across all 12
    /// GRF components per step).
    pub r_diag: f64,
    /// **(Step B)** If `true`, the MPC's input is extended to 24 dims per
    /// step `[F (12); Δr (12)]`, where `Δr_i` is an additive foot-offset
    /// the controller asks the swing planner to apply on the next
    /// touchdown of leg `i` (body frame, metres). Enables true
    /// closed-loop foot placement — the MPC can ask the foot to land
    /// further outboard to catch a lateral disturbance, mirroring
    /// `legged_control`'s OCS2-based foot landing planning.
    ///
    /// When `false` (default), the MPC behaves identically to the
    /// 12-input version and `MpcSolution.foot_offsets_*` are zero.
    /// The `Δr × F` term in `ω̇` is bilinear, so the QP linearizes
    /// around an iterate `F^*` (taken from the previous MPC solve or
    /// from a gravity-balanced baseline on the first solve).
    pub enable_foot_offset: bool,
    /// **(Step B)** Cost weight on each foot-offset component (m).
    /// Default: 0.5, which keeps the regularizer comparable to the
    /// state-tracking budget while still letting the MPC commit to a
    /// few-cm offset under disturbance.
    pub r_diag_foot_offset: f64,
    /// **(Step B)** Hard bound on `|Δr_i|` per axis (m). Default 0.08 —
    /// roughly half a stride at default trot params; large enough for
    /// useful corrections, small enough to keep the IK in workspace.
    pub max_foot_offset_m: f64,
}

impl Default for SrbdMpcConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 10,
            dt_per_step: 0.030,
            mass_kg: 9.0,
            inertia_diag_body: Vector3::new(0.07, 0.26, 0.242),
            friction_mu: 0.5,
            max_normal_force: 200.0,
            // Cheetah-3 baseline (Di Carlo 2018 §V) with lateral /
            // yaw stabilisation weights bumped from the original
            // forward-only tuning. Without these the MPC ignores
            // lateral position drift (q_diag[4] = 0) and yaw drift
            // (q_diag[2] = 0.5), which under MuJoCo's open-loop
            // friction asymmetries produces ~100° of yaw drift over
            // 5 s of forward walk (visible in
            // `tests/integration_walk.rs` straight-walk benchmarks).
            //
            // Settings:
            //   θ_x, θ_y     : 25.0 (Cheetah-3 default — roll/pitch upright)
            //   θ_z (yaw)    : 25.0 ←  was 0.5; matches roll/pitch so
            //                          yaw drift gets penalised equally
            //   p_x          : 0.0  (forward position is integrated by
            //                          velocity command — no fixed target)
            //   p_y          : 20.0 ←  was 0.0; small lateral-position
            //                          regulariser that pulls drifting
            //                          body back to commanded line. A
            //                          D3.3.8 attempt to bump this to
            //                          30/50 broke `integration_walk_
            //                          lateral_mpc_no_wbc` (over-
            //                          correction → fall), so 20 is the
            //                          empirical ceiling on namiashi.
            //   p_z          : 50.0 (height-hold — Cheetah default)
            //   ω_z (yaw rate): 5.0 ←  was 0.5; reinforces θ_z
            q_diag: [
                // θ_x, θ_y, θ_z
                25.0, 25.0, 50.0,
                // p_x, p_y, p_z
                0.0, 20.0, 50.0,
                // ω_x, ω_y, ω_z
                0.5, 0.5, 10.0,
                // v_x, v_y, v_z
                1.0, 1.0, 1.0,
                // g
                0.0,
            ],
            // Input cost. Di Carlo 2018 §V uses 1e-6 for Cheetah-3,
            // but with smaller robots (~2.4 kg namiashi) the
            // dynamics constraint has a wide null space, so an
            // essentially-zero r_diag lets clarabel's interior point
            // pick wildly different optima on consecutive solves
            // (13 → 80 N at static stand). 1e-3 narrows the optimal
            // set without distorting the cost balance — Cheetah-3
            // hover test still passes (Σf_z ≈ m·g = 88 N within 15%)
            // and namiashi static stand settles at GRF ≈ 25 N
            // (m·g = 23.5 N expected).
            r_diag: 1e-3,
            // Step B: foot-offset extension is opt-in. The 12-input
            // pre-Step-B path stays the default so existing tests are
            // unaffected; flip via `cfg.enable_foot_offset = true`.
            enable_foot_offset: false,
            r_diag_foot_offset: 0.5,
            max_foot_offset_m: 0.08,
        }
    }
}

/// Minimal contact schedule for the MPC: per-leg per-step boolean.
/// `is_stance[leg][k] == true` ⇒ that foot is in stance at step k and
/// can apply force.
#[derive(Clone, Debug)]
pub struct ContactSchedule {
    /// `[FL, FR, RL, RR]` × `[step 0, 1, …, N-1]`.
    pub is_stance: [Vec<bool>; 4],
}

impl ContactSchedule {
    /// All four feet in stance for every step (used as a sanity-check
    /// fallback / initial guess when phase data isn't available).
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

/// Reference trajectory the MPC tries to track. One entry per horizon
/// step (length must match `cfg.horizon_steps`).
#[derive(Clone, Debug)]
pub struct ReferenceTrajectory {
    pub states: Vec<SrbdState>,
}

impl ReferenceTrajectory {
    /// Constant-state reference (everything stays at `s` over the
    /// horizon). Useful for hover / standing.
    pub fn constant(s: SrbdState, horizon_steps: usize) -> Self {
        Self { states: vec![s; horizon_steps] }
    }

    /// Constant-velocity reference: integrate position + yaw forward
    /// from `s_now` at velocity `(vx, vy, wz)` (body frame, expressed
    /// here in world frame for simplicity — the controller calls this
    /// after rotating into world).
    pub fn from_constant_velocity(
        s_now: SrbdState,
        v_world: Vector3<f64>,
        wz: f64,
        cfg: &SrbdMpcConfig,
    ) -> Self {
        let mut states = Vec::with_capacity(cfg.horizon_steps);
        let mut s = s_now;
        s.linear_velocity = v_world;
        s.angular_velocity = Vector3::new(0.0, 0.0, wz);
        for k in 0..cfg.horizon_steps {
            let t = (k + 1) as f64 * cfg.dt_per_step;
            let mut sk = s;
            sk.position = s_now.position + v_world * t;
            sk.orientation_rpy.z = s_now.orientation_rpy.z + wz * t;
            states.push(sk);
        }
        Self { states }
    }
}

/// Per-foot CoM-relative offset over the horizon.
/// `r[leg][k]` = world-frame vector from CoM (at step k) to that foot
/// (also at step k). For trotting at moderate speeds the foot is
/// close to the body's projection, so a constant approximation
/// (`[hip_offset; … nominal_z]`) is acceptable for a 300 ms horizon.
#[derive(Clone, Debug)]
pub struct FootOffsets {
    pub r: [Vec<Vector3<f64>>; 4],
}

impl FootOffsets {
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

/// Result of [`SrbdMpc::solve`].
#[derive(Clone, Debug)]
pub struct MpcSolution {
    /// GRF for each leg at the **first** horizon step (world frame, N).
    /// This is what the host would feed to a torque-control mapper.
    pub grfs_first_step: [Vector3<f64>; 4],
    /// GRF for every leg at every horizon step. Useful for diagnostic
    /// plots; the full vector is what clarabel returned.
    pub grfs_all_steps: Vec<[Vector3<f64>; 4]>,
    /// **(Step B)** Foot offset `Δr_i` per leg at the first horizon step
    /// (body frame, m). Zero when [`SrbdMpcConfig::enable_foot_offset`]
    /// is `false`. The controller's footstep planner adds this to the
    /// Raibert touchdown target to act on the MPC's plan.
    pub foot_offsets_first_step: [Vector3<f64>; 4],
    /// **(Step B)** Foot offsets at every horizon step.
    pub foot_offsets_all_steps: Vec<[Vector3<f64>; 4]>,
    /// **Predicted body states** at every horizon step `k = 1..=n`
    /// (i.e. step 0 is the *current* state, step 1 is one `dt_per_step`
    /// in the future, …, step n is the end of the horizon).
    ///
    /// Computed from the closed-form propagation `x_{k+1} = A_k·x_k +
    /// B_k·u_k` using the linearised dynamics + the QP-optimal `U`.
    /// Used by [`crate::mpc_reference`] to plan joint-space references
    /// (q*/q̇*) per horizon step — the missing piece that lets the
    /// WBC swing_leg task share its target with Position-PD instead
    /// of fighting it (the typical OCS2 NMPC role in legged_control).
    pub predicted_body_states: Vec<SrbdState>,
    /// Solver-reported objective value at the optimum.
    pub objective: f64,
    /// Whether clarabel converged (otherwise `grfs_*` are best-effort
    /// values and may violate constraints).
    pub solved: bool,
}

/// SRBD physics: predict the floating-base linear and angular
/// acceleration **the MPC's GRFs would produce** at the current state.
///
/// Mirrors `legged_control`'s `formulateBaseAccelTask`, which derives
/// `a_base_des` from the OCS2 NMPC's centroidal momentum rate via
/// `A_base⁻¹ · momentum_rate`. We don't have OCS2's centroidal model
/// here, but for a Single Rigid Body the equivalent reduces to the
/// SRBD's own continuous dynamics:
///
/// ```text
/// p̈_world = (Σ f_GRF_i) / m  +  g_world          (Newton)
/// α_world = I_world⁻¹ · (Σ (r_i − p_body) × f_GRF_i  −  ω × (I_world · ω))
/// ```
///
/// The result is the WBC's `a_base_des` reference: feeding it directly
/// (in place of a hand-tuned PD on body velocity) makes the WBC track
/// **the MPC's own predicted body motion**, eliminating the systemic
/// MPC-vs-WBC mismatch that otherwise excites the floating base into
/// joint-velocity divergence.
///
/// Returns `(a_lin_world, alpha_world)` — both world frame. The host
/// rotates them into the WBC's body-frame motion subspace.
///
/// Approximations:
/// - `I_world ≈ R(rpy) · diag(I_body) · R(rpy)ᵀ` (full Euler rotation,
///   matching the body's actual orientation rather than SRBD's
///   yaw-only linearisation, since the WBC sees the true tilt).
/// - Foot positions are taken in world frame (caller passes them).
pub fn predicted_base_accel_world(
    cfg: &SrbdMpcConfig,
    state: &SrbdState,
    grfs: &[Vector3<f64>; 4],
    foot_positions_world: &[Vector3<f64>; 4],
) -> (Vector3<f64>, Vector3<f64>) {
    use nalgebra::{Matrix3, Rotation3};
    let g_world = Vector3::new(0.0, 0.0, -9.81);

    // Linear: p̈ = Σf/m + g (g points down → static stand has Σf = m·g·ẑ → p̈ = 0).
    let total_f: Vector3<f64> = grfs.iter().sum();
    let a_lin_world = total_f / cfg.mass_kg.max(1e-9) + g_world;

    // Angular: α = I_world⁻¹ · (Σ r_i × f_i  −  ω × (I·ω))
    let r_world_body = Rotation3::from_euler_angles(
        state.orientation_rpy.x,
        state.orientation_rpy.y,
        state.orientation_rpy.z,
    );
    let i_body = Matrix3::from_diagonal(&cfg.inertia_diag_body);
    let i_world = r_world_body.matrix() * i_body * r_world_body.matrix().transpose();
    // SrbdState stores ω in body frame; the τ × f cross-product needs world frame.
    let omega_world = r_world_body * state.angular_velocity;
    let body_pos = state.position;
    let mut tau_world = Vector3::zeros();
    for slot in 0..4 {
        let r = foot_positions_world[slot] - body_pos;
        tau_world += r.cross(&grfs[slot]);
    }
    let i_omega = i_world * omega_world;
    let coriolis = omega_world.cross(&i_omega);
    let alpha_world = i_world
        .try_inverse()
        .unwrap_or_else(Matrix3::identity)
        * (tau_world - coriolis);

    (a_lin_world, alpha_world)
}

/// Stateful MPC solver. Holds the cached `SrbdMpcConfig` so the host
/// can rebuild the QP each tick without re-allocating tuning data.
#[derive(Clone, Debug)]
pub struct SrbdMpc {
    cfg: SrbdMpcConfig,
}

impl SrbdMpc {
    pub fn new(cfg: SrbdMpcConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &SrbdMpcConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: SrbdMpcConfig) {
        self.cfg = cfg;
    }

    /// Build the SRBD QP and solve it. The host passes:
    /// - `state_now`: current SRBD state (yaw used for rotation linearisation)
    /// - `reference`: per-step desired state over the horizon
    /// - `contact`: per-leg per-step stance flag
    /// - `feet`: per-leg per-step CoM-to-foot offsets (world frame)
    ///
    /// Returns [`MpcSolution`] with the GRFs to apply at the first
    /// step; subsequent steps are returned for diagnostic plotting
    /// but are NOT applied (Receding Horizon — only first step is
    /// committed each tick).
    pub fn solve(
        &self,
        state_now: SrbdState,
        reference: &ReferenceTrajectory,
        contact: &ContactSchedule,
        feet: &FootOffsets,
    ) -> MpcSolution {
        let n = self.cfg.horizon_steps;
        assert_eq!(reference.states.len(), n, "ref length mismatch");
        for leg in 0..4 {
            assert_eq!(contact.is_stance[leg].len(), n);
            assert_eq!(feet.r[leg].len(), n);
        }

        // ── Build per-step continuous-time A_c, B_c, then discretise ──
        //
        // A_c is (mostly) time-invariant (only the yaw-rotation block
        // depends on ψ_k from the reference). B_c depends on r_i,k
        // and the inertia. Discretise via Euler step.
        //
        // Step B: when foot-offset extension is enabled, the bilinear
        // `Δr × F` term is linearized around a GRF iterate F^*. As a
        // single-iteration SQP, we use the gravity-balanced GRF
        // distributed across stance legs (= what hover would need).
        // This is sufficient for small disturbances. A full SQP outer
        // loop (re-solve with updated F^*) is left for later.
        let grf_hover_per_stance = if self.cfg.enable_foot_offset {
            let mg = self.cfg.mass_kg * 9.81;
            Some(mg)
        } else {
            None
        };

        let mut a_d_per_step: Vec<DMatrix<f64>> = Vec::with_capacity(n);
        let mut b_d_per_step: Vec<DMatrix<f64>> = Vec::with_capacity(n);
        for k in 0..n {
            let psi_ref = reference.states[k].orientation_rpy.z;
            let r_per_leg = [feet.r[0][k], feet.r[1][k], feet.r[2][k], feet.r[3][k]];
            let stance = [
                contact.is_stance[0][k],
                contact.is_stance[1][k],
                contact.is_stance[2][k],
                contact.is_stance[3][k],
            ];
            let grf_iterate = if let Some(mg) = grf_hover_per_stance {
                let n_stance = stance.iter().filter(|&&s| s).count();
                let f_z_per = if n_stance > 0 { mg / n_stance as f64 } else { 0.0 };
                [
                    Vector3::new(0.0, 0.0, if stance[0] { f_z_per } else { 0.0 }),
                    Vector3::new(0.0, 0.0, if stance[1] { f_z_per } else { 0.0 }),
                    Vector3::new(0.0, 0.0, if stance[2] { f_z_per } else { 0.0 }),
                    Vector3::new(0.0, 0.0, if stance[3] { f_z_per } else { 0.0 }),
                ]
            } else {
                [Vector3::zeros(); 4]
            };
            let (a_c, b_c) = self.continuous_dynamics(psi_ref, &r_per_leg, &stance, &grf_iterate);
            // Forward Euler discretisation.
            let mut a_d = DMatrix::<f64>::identity(13, 13);
            a_d += &a_c * self.cfg.dt_per_step;
            let b_d = b_c * self.cfg.dt_per_step;
            a_d_per_step.push(a_d);
            b_d_per_step.push(b_d);
        }

        // ── Lifted dynamics: X = A_x x_0 + B_u U  ───────────────────
        //
        // Where X = [x_1; …; x_N] (13N × 1), U = [u_0; …; u_{N-1}] (12N × 1)
        //
        //   A_x[k,:] = (A_{k-1} … A_0) · x_0
        //   B_u[k,j] = (A_{k-1} … A_{j+1}) · B_j   for j ≤ k-1, else 0
        //
        // Build by accumulating the partial product as we walk forward.
        let nx = 13;
        let nu = if self.cfg.enable_foot_offset { 24 } else { 12 };
        let mut a_x = DMatrix::<f64>::zeros(nx * n, nx);
        let mut b_u = DMatrix::<f64>::zeros(nx * n, nu * n);

        // Cache running products A_{k} · A_{k-1} · … so each row's
        // contribution is O(1) instead of recomputing the chain.
        let mut prod = DMatrix::<f64>::identity(nx, nx); // initially I (no As multiplied)
        for k in 0..n {
            // x_{k+1} = A_k · x_k + B_k · u_k.
            // Updated product covers A_k · A_{k-1} · … · A_0.
            prod = &a_d_per_step[k] * &prod;
            // Row k of A_x = product so far.
            a_x.view_mut((k * nx, 0), (nx, nx)).copy_from(&prod);
            // Row k of B_u: contribution from each input u_j, j ≤ k.
            // The mapping is B_u[k,j] = (A_k · A_{k-1} · … · A_{j+1}) · B_j.
            // We accumulate from j = k down to 0 so the prefix product
            // can be reused.
            let mut tail = DMatrix::<f64>::identity(nx, nx);
            for j in (0..=k).rev() {
                let block = &tail * &b_d_per_step[j];
                b_u.view_mut((k * nx, j * nu), (nx, nu)).copy_from(&block);
                if j > 0 {
                    tail = &tail * &a_d_per_step[j];
                }
            }
        }

        // ── Cost: J = (X − X_ref)^T Q_block (X − X_ref) + U^T R_block U
        //       = U^T (B_u^T Q B_u + R) U + 2 (B_u^T Q (A_x x_0 − X_ref))^T U + const
        //       = ½ U^T (2(B_u^T Q B_u + R)) U + (2 B_u^T Q (A_x x_0 − X_ref))^T U
        //
        // → P = 2 (B_u^T Q B_u + R), q = 2 B_u^T Q (A_x x_0 − X_ref)
        let mut q_block = DMatrix::<f64>::zeros(nx * n, nx * n);
        for k in 0..n {
            for i in 0..nx {
                q_block[(k * nx + i, k * nx + i)] = self.cfg.q_diag[i];
            }
        }
        let mut r_block = DMatrix::<f64>::zeros(nu * n, nu * n);
        for i in 0..(nu * n) {
            let leg_input_idx = i % nu;
            r_block[(i, i)] = if leg_input_idx < 12 {
                // GRF entries (12 per step).
                self.cfg.r_diag
            } else {
                // Foot-offset entries (only present when enable_foot_offset).
                self.cfg.r_diag_foot_offset
            };
        }

        let x_ref = {
            let mut v = DVector::<f64>::zeros(nx * n);
            for k in 0..n {
                let s = reference.states[k].to_vec();
                v.rows_mut(k * nx, nx).copy_from(&s);
            }
            v
        };
        let x_now = state_now.to_vec();
        let drift = &a_x * &x_now - &x_ref; // A_x x_0 − X_ref

        let p_dense = 2.0 * (b_u.transpose() * &q_block * &b_u + &r_block);
        let q_vec = 2.0 * (b_u.transpose() * &q_block * &drift);

        // ── Constraints ─────────────────────────────────────────────
        //
        // Per leg per step:
        //   stance: 0 ≤ f_z ≤ f_max
        //           |f_x| ≤ μ f_z   (linearised friction pyramid)
        //           |f_y| ≤ μ f_z
        //   swing:  f_x = f_y = f_z = 0  (equality)
        //
        // Conventions for clarabel:
        //   ZeroConeT(m)         : A·x − b = 0 over the first m rows
        //   NonnegativeConeT(m)  : A·x − b ≤ 0 over the next m rows
        //
        // We stack equalities first, then inequalities.
        let (a_csc, b_vec, cones) = build_constraints(&self.cfg, contact, n, nu);

        // ── clarabel solve ──────────────────────────────────────────
        let p_csc = dense_to_csc_upper(&p_dense);
        let q_slice: Vec<f64> = q_vec.iter().copied().collect();
        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.max_iter = 50;
        let mut solver = match DefaultSolver::new(
            &p_csc,
            &q_slice,
            &a_csc,
            &b_vec,
            &cones,
            settings,
        ) {
            Ok(s) => s,
            Err(_) => {
                // Constructor failed — return zero GRFs marked as not solved.
                return MpcSolution {
                    grfs_first_step: [Vector3::zeros(); 4],
                    grfs_all_steps: vec![[Vector3::zeros(); 4]; n],
                    foot_offsets_first_step: [Vector3::zeros(); 4],
                    foot_offsets_all_steps: vec![[Vector3::zeros(); 4]; n],
                    // Best-effort predicted state on solver failure:
                    // hold current state for the entire horizon.
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
            let g = [
                Vector3::new(u_opt[base], u_opt[base + 1], u_opt[base + 2]),
                Vector3::new(u_opt[base + 3], u_opt[base + 4], u_opt[base + 5]),
                Vector3::new(u_opt[base + 6], u_opt[base + 7], u_opt[base + 8]),
                Vector3::new(u_opt[base + 9], u_opt[base + 10], u_opt[base + 11]),
            ];
            grfs_all_steps.push(g);
        }

        // Decode predicted body states by propagating the SRBD
        // dynamics with the QP-optimal U: X = A_x · x_0 + B_u · U.
        // X is laid out [x_1; x_2; …; x_n] (each 13-dim).
        let u_dvec = DVector::from_vec(u_opt.clone());
        let x_horizon = &a_x * &x_now + &b_u * &u_dvec;
        let mut predicted_body_states = Vec::with_capacity(n);
        for k in 0..n {
            let row0 = k * nx;
            predicted_body_states.push(SrbdState {
                orientation_rpy: Vector3::new(
                    x_horizon[row0],
                    x_horizon[row0 + 1],
                    x_horizon[row0 + 2],
                ),
                position: Vector3::new(
                    x_horizon[row0 + 3],
                    x_horizon[row0 + 4],
                    x_horizon[row0 + 5],
                ),
                angular_velocity: Vector3::new(
                    x_horizon[row0 + 6],
                    x_horizon[row0 + 7],
                    x_horizon[row0 + 8],
                ),
                linear_velocity: Vector3::new(
                    x_horizon[row0 + 9],
                    x_horizon[row0 + 10],
                    x_horizon[row0 + 11],
                ),
            });
        }

        // Step B: when foot-offset extension is enabled, the QP returns
        // an additional 12 components per step (Δr_x/y/z per leg). Decode
        // them too, otherwise leave zeros.
        let (foot_offsets_first_step, foot_offsets_all_steps) =
            if self.cfg.enable_foot_offset {
                let mut all: Vec<[Vector3<f64>; 4]> = Vec::with_capacity(n);
                for k in 0..n {
                    // Layout per step: [F_FL F_FR F_RL F_RR | Δr_FL Δr_FR Δr_RL Δr_RR].
                    let base = k * (12 + 12) + 12;
                    all.push([
                        Vector3::new(u_opt[base], u_opt[base + 1], u_opt[base + 2]),
                        Vector3::new(u_opt[base + 3], u_opt[base + 4], u_opt[base + 5]),
                        Vector3::new(u_opt[base + 6], u_opt[base + 7], u_opt[base + 8]),
                        Vector3::new(u_opt[base + 9], u_opt[base + 10], u_opt[base + 11]),
                    ]);
                }
                (all[0], all)
            } else {
                ([Vector3::zeros(); 4], vec![[Vector3::zeros(); 4]; n])
            };

        MpcSolution {
            grfs_first_step: grfs_all_steps[0],
            grfs_all_steps,
            foot_offsets_first_step,
            foot_offsets_all_steps,
            predicted_body_states,
            objective,
            solved,
        }
    }

    /// Continuous-time A and B matrices at one horizon step. See
    /// module docs for the full state / input layout.
    ///
    /// `grf_iterate` is the GRF reference used to linearize the bilinear
    /// term `Δr_i × F_i` when `enable_foot_offset = true`. For stance
    /// legs, the offset columns in B are `I⁻¹ · [F^*_i]×`. For swing
    /// legs (`stance[i] = false`) the offset columns stay zero
    /// (Δr has no effect when F = 0). When `enable_foot_offset = false`
    /// the input dimension is 12 (GRFs only) and `grf_iterate` is
    /// unused.
    fn continuous_dynamics(
        &self,
        psi_ref: f64,
        r_per_leg: &[Vector3<f64>; 4],
        stance: &[bool; 4],
        grf_iterate: &[Vector3<f64>; 4],
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let nx = 13;
        let nu = if self.cfg.enable_foot_offset { 24 } else { 12 };
        let mut a = DMatrix::<f64>::zeros(nx, nx);
        let mut b = DMatrix::<f64>::zeros(nx, nu);

        // θ̇ = R_z(ψ_ref)^T · ω
        // For yaw-only linearisation, R_z(ψ)^T = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        let (s, c) = psi_ref.sin_cos();
        a[(0, 6)] = c;
        a[(0, 7)] = s;
        a[(1, 6)] = -s;
        a[(1, 7)] = c;
        a[(2, 8)] = 1.0;

        // ṗ = v
        a[(3, 9)] = 1.0;
        a[(4, 10)] = 1.0;
        a[(5, 11)] = 1.0;

        // v̇_z has the gravity column (last state).
        // d(v_z)/dt = ... + g  → entry at (11, 12) = 1
        a[(11, 12)] = 1.0;

        // World-frame inertia approximated by body-frame diagonal
        // rotated by yaw. For yaw-only linearisation this is
        //   I_world = R_z(ψ) · diag(I_body) · R_z(ψ)^T
        // Compute its inverse component-wise (still sparse).
        let i_body_diag = self.cfg.inertia_diag_body;
        let i_world = world_inertia_yaw_only(psi_ref, i_body_diag);
        let i_inv = invert_3x3(&i_world);

        // ω̇ = I_world^{-1} · Σ_i [r_i]× · f_i   (per leg)
        // v̇  = (1/m) · Σ_i f_i
        for leg in 0..4 {
            let col_base = leg * 3;
            if !stance[leg] {
                // No force from this leg → leave columns zero. Equality
                // constraints below will pin the QP variables to 0.
                continue;
            }
            // Cross-product matrix [r_i]×
            let r = r_per_leg[leg];
            let r_cross = skew_symmetric(&r);
            let m_block = &i_inv * &r_cross;
            for i in 0..3 {
                for j in 0..3 {
                    b[(6 + i, col_base + j)] = m_block[(i, j)];
                }
            }
            // Linear velocity contribution: 1/m on diag.
            for i in 0..3 {
                b[(9 + i, col_base + i)] = 1.0 / self.cfg.mass_kg;
            }
        }

        // Step B: bilinear `Δr_i × F_i` contributes to ω̇. Linearize as
        // `Δr_i × F^*_i` where F^* is the previous-iteration GRF (passed
        // in via `grf_iterate`). The Δr columns sit at offsets 12..24
        // in the extended input.
        if self.cfg.enable_foot_offset {
            for leg in 0..4 {
                let col_base = 12 + leg * 3;
                if !stance[leg] {
                    // F = 0 on this leg, so Δr × F = 0 too — leave zero.
                    continue;
                }
                // [Δr]× · F = -[F]× · Δr, i.e. the Jacobian wrt Δr is
                // `-[F^*]×`. (Equivalent up to sign convention; the
                // skew-symmetric matrix [F^*]× of F^* is used directly
                // below — the SQP iterate sign drops out in the cost
                // minimization, what matters is that the partial
                // derivative magnitude is right.)
                let f_star = grf_iterate[leg];
                let f_cross = skew_symmetric(&(-f_star));
                let m_block = &i_inv * &f_cross;
                for i in 0..3 {
                    for j in 0..3 {
                        b[(6 + i, col_base + j)] = m_block[(i, j)];
                    }
                }
                // Δr has no direct effect on v̇ (only on moment arm) so
                // rows 9..12 stay zero in the offset columns.
            }
        }

        (a, b)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn skew_symmetric(v: &Vector3<f64>) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(3, 3);
    m[(0, 1)] = -v.z;
    m[(0, 2)] = v.y;
    m[(1, 0)] = v.z;
    m[(1, 2)] = -v.x;
    m[(2, 0)] = -v.y;
    m[(2, 1)] = v.x;
    m
}

fn world_inertia_yaw_only(psi: f64, i_body_diag: Vector3<f64>) -> DMatrix<f64> {
    // R = [[c,-s,0],[s,c,0],[0,0,1]]
    // I_world = R · diag(Ix, Iy, Iz) · R^T
    let (s, c) = psi.sin_cos();
    let ix = i_body_diag.x;
    let iy = i_body_diag.y;
    let iz = i_body_diag.z;
    let mut m = DMatrix::<f64>::zeros(3, 3);
    m[(0, 0)] = c * c * ix + s * s * iy;
    m[(0, 1)] = c * s * (ix - iy);
    m[(1, 0)] = c * s * (ix - iy);
    m[(1, 1)] = s * s * ix + c * c * iy;
    m[(2, 2)] = iz;
    m
}

fn invert_3x3(m: &DMatrix<f64>) -> DMatrix<f64> {
    // Use nalgebra's built-in inverse on a fixed-size matrix copy.
    let m3 = nalgebra::Matrix3::from_iterator(m.iter().copied());
    let inv = m3
        .try_inverse()
        .unwrap_or_else(nalgebra::Matrix3::identity);
    DMatrix::from_iterator(3, 3, inv.iter().copied())
}

/// Convert a dense symmetric PSD matrix into clarabel's CSC upper-
/// triangular representation. clarabel reads only the upper triangle
/// (column-major) and assumes symmetry.
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

/// Build the constraint matrix A and bound vector b for the QP. The
/// stacking order is:
///   1. equality rows for swing legs (f = 0)
///   2. inequality rows for stance legs:
///      - f_z ≥ 0           ⇒ -f_z ≤ 0
///      - f_z ≤ f_max       ⇒ f_z − f_max ≤ 0
///      - f_x − μ·f_z ≤ 0
///      - −f_x − μ·f_z ≤ 0
///      - f_y − μ·f_z ≤ 0
///      - −f_y − μ·f_z ≤ 0
fn build_constraints(
    cfg: &SrbdMpcConfig,
    contact: &ContactSchedule,
    n: usize,
    nu: usize,
) -> (CscMatrix<f64>, Vec<f64>, Vec<clarabel::solver::SupportedConeT<f64>>) {
    let total_vars = nu * n;
    let mu = cfg.friction_mu;
    let f_max = cfg.max_normal_force;
    let offset_enabled = cfg.enable_foot_offset;
    let r_offset_max = cfg.max_foot_offset_m;

    // Pre-count rows for sizing.
    let mut n_eq = 0;
    let mut n_ineq = 0;
    for k in 0..n {
        for leg in 0..4 {
            if contact.is_stance[leg][k] {
                let mut count = 4; // friction pyramid
                count += 1; // f_z ≥ 0
                if f_max > 0.0 {
                    count += 1; // f_z ≤ f_max
                }
                n_ineq += count;
            } else {
                n_eq += 3; // f_x = f_y = f_z = 0
            }
        }
        if offset_enabled {
            // Δr bounds: |Δr_ij| ≤ r_offset_max for each (leg, axis).
            // 24 inequalities per step (12 components × 2 sides).
            n_ineq += 24;
        }
    }

    let n_rows = n_eq + n_ineq;
    let mut a_dense = DMatrix::<f64>::zeros(n_rows, total_vars);
    let mut b_vec = vec![0.0; n_rows];
    let mut row = 0;

    // Equality rows first
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
            // -f_z ≤ 0  ⇒  row[col_z] = -1, b = 0
            a_dense[(row, col_z)] = -1.0;
            b_vec[row] = 0.0;
            row += 1;
            // f_z ≤ f_max  ⇒  row[col_z] = 1, b = f_max
            if f_max > 0.0 {
                a_dense[(row, col_z)] = 1.0;
                b_vec[row] = f_max;
                row += 1;
            }
            // f_x − μ f_z ≤ 0
            a_dense[(row, col_x)] = 1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
            // -f_x − μ f_z ≤ 0
            a_dense[(row, col_x)] = -1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
            // f_y − μ f_z ≤ 0
            a_dense[(row, col_y)] = 1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
            // -f_y − μ f_z ≤ 0
            a_dense[(row, col_y)] = -1.0;
            a_dense[(row, col_z)] = -mu;
            row += 1;
        }
        // Step B: Δr workspace bounds. Δr_ij sits at column
        // `k * nu + 12 + leg * 3 + axis`. We emit one pair (+/-) per
        // axis, 12 components × 2 sides = 24 per step. Applied to
        // both stance and swing legs (swing legs' Δr has no dynamics
        // effect but is constrained to stay in workspace for sanity).
        if offset_enabled {
            for leg in 0..4 {
                for axis in 0..3 {
                    let col = k * nu + 12 + leg * 3 + axis;
                    // Δr ≤ r_offset_max
                    a_dense[(row, col)] = 1.0;
                    b_vec[row] = r_offset_max;
                    row += 1;
                    // -Δr ≤ r_offset_max  (= Δr ≥ -r_offset_max)
                    a_dense[(row, col)] = -1.0;
                    b_vec[row] = r_offset_max;
                    row += 1;
                }
            }
        }
    }

    let a_csc = dense_to_csc_full(&a_dense);
    let mut cones: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();
    if n_eq > 0 {
        cones.push(ZeroConeT(n_eq));
    }
    if n_ineq > 0 {
        cones.push(NonnegativeConeT(n_ineq));
    }
    (a_csc, b_vec, cones)
}

/// Convert a dense matrix into clarabel's CSC format (column-major
/// nonzero lists). Unlike `dense_to_csc_upper`, this stores **all**
/// nonzeros — used for the constraint matrix `A` which is general.
fn dense_to_csc_full(a: &DMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for i in 0..m {
            let v = a[(i, j)];
            if v.abs() > 1e-12 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

// ─── Convenience helpers ──────────────────────────────────────────────

/// Map per-leg outputs to the canonical [FL, FR, RL, RR] slot.
pub const LEG_SLOTS: [LegId; 4] = [LegId::FL, LegId::FR, LegId::RL, LegId::RR];

// Trait bound to satisfy clarabel's generics — sanity check that f64
// satisfies it; never actually called.
#[allow(dead_code)]
fn _clarabel_compat<T: FloatT>() {}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hover_state() -> SrbdState {
        let mut s = SrbdState::default();
        s.position.z = 0.30;
        s
    }

    fn nominal_feet() -> [Vector3<f64>; 4] {
        // Cheetah-class hip offsets, foot directly below hip
        // (z = -nominal stance height).
        [
            Vector3::new(0.18, 0.10, -0.30),
            Vector3::new(0.18, -0.10, -0.30),
            Vector3::new(-0.18, 0.10, -0.30),
            Vector3::new(-0.18, -0.10, -0.30),
        ]
    }

    /// Static stand: GRFs that exactly balance gravity (Σf_z = m·g, all
    /// horizontal components zero) must yield zero predicted base
    /// acceleration. This is the key sanity check for
    /// `predicted_base_accel_world`: the function is the heart of the
    /// MPC→WBC reference pipeline and must agree with Newton's laws on
    /// the trivial case before we trust the rest.
    #[test]
    fn predicted_base_accel_zero_under_balanced_grfs() {
        let cfg = SrbdMpcConfig::default();
        let mut s = SrbdState::default();
        s.position.z = 0.30;
        let feet_world = [
            Vector3::new(0.18, 0.10, 0.0),
            Vector3::new(0.18, -0.10, 0.0),
            Vector3::new(-0.18, 0.10, 0.0),
            Vector3::new(-0.18, -0.10, 0.0),
        ];
        // Balance gravity exactly: each foot pushes up m·g/4.
        let f_each = cfg.mass_kg * 9.81 / 4.0;
        let grfs = [Vector3::new(0.0, 0.0, f_each); 4];

        let (a_lin, a_ang) = predicted_base_accel_world(&cfg, &s, &grfs, &feet_world);
        for i in 0..3 {
            assert!(a_lin[i].abs() < 1e-9, "a_lin[{i}] = {} should be ≈0", a_lin[i]);
            assert!(a_ang[i].abs() < 1e-9, "a_ang[{i}] = {} should be ≈0", a_ang[i]);
        }
    }

    /// Asymmetric vertical load: front feet carry double the rear feet.
    /// Linear z-acceleration must still match Σf/m + g, and the
    /// resulting torque about the body y-axis (pitch) must be non-zero
    /// in the expected direction (front-up → nose-up pitch torque).
    #[test]
    fn predicted_base_accel_pitch_torque_sign() {
        let cfg = SrbdMpcConfig::default();
        let s = SrbdState {
            position: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let feet_world = [
            Vector3::new(0.18, 0.10, 0.0),  // FL (front)
            Vector3::new(0.18, -0.10, 0.0), // FR (front)
            Vector3::new(-0.18, 0.10, 0.0), // RL (rear)
            Vector3::new(-0.18, -0.10, 0.0),// RR (rear)
        ];
        let f_total = cfg.mass_kg * 9.81;
        let f_front = f_total * 2.0 / 6.0; // 2/6 each front
        let f_rear = f_total * 1.0 / 6.0;  // 1/6 each rear
        let grfs = [
            Vector3::new(0.0, 0.0, f_front),
            Vector3::new(0.0, 0.0, f_front),
            Vector3::new(0.0, 0.0, f_rear),
            Vector3::new(0.0, 0.0, f_rear),
        ];

        let (a_lin, a_ang) = predicted_base_accel_world(&cfg, &s, &grfs, &feet_world);
        // Total Σf_z = 2·f_front + 2·f_rear = 2·(2/6 + 1/6)·m·g = m·g → a_lin_z = 0.
        assert!(a_lin.z.abs() < 1e-9, "a_lin_z = {} should be ≈0", a_lin.z);
        // Torque about body y-axis: front feet pull body forward-up, rear
        // pull less. Cross product r × f for a front foot at +x with f_z > 0
        // yields τ in -y direction; rear at -x yields +y direction. Net is
        // (f_rear − f_front) summed pairs → negative y (nose-down pitch
        // torque) because front feet push harder. So α_y < 0.
        assert!(a_ang.y < -1e-3, "α_y = {} should be < 0 (nose-down pitch)", a_ang.y);
    }

    /// Stationary all-stance hover: total z-force across all four feet
    /// must equal `m·g` (≈ 88 N at 9 kg) and individual feet should
    /// each carry ~m·g / 4. Sanity check that the dynamics are correct
    /// and the cost weights aren't hopelessly off.
    #[test]
    fn hover_distributes_weight_evenly() {
        let cfg = SrbdMpcConfig::default();
        let mpc = SrbdMpc::new(cfg.clone());
        let s = hover_state();
        let n = cfg.horizon_steps;
        let reference = ReferenceTrajectory::constant(s, n);
        let contact = ContactSchedule::all_stance(n);
        let feet = FootOffsets::constant_per_leg(nominal_feet(), n);

        let sol = mpc.solve(s, &reference, &contact, &feet);
        assert!(sol.solved, "hover QP must converge");
        let total_fz: f64 = sol.grfs_first_step.iter().map(|f| f.z).sum();
        let weight = cfg.mass_kg * 9.81;
        assert!(
            (total_fz - weight).abs() < 0.15 * weight,
            "Σf_z = {total_fz} should be ≈ m·g = {weight}",
        );
        // Symmetry: nominal feet are placed symmetrically so the four
        // f_z entries should each be ≈ weight/4. Allow 30% tolerance —
        // the cost weights bias the symmetry slightly.
        let avg = total_fz / 4.0;
        for f in &sol.grfs_first_step {
            assert!(
                (f.z - avg).abs() < 0.3 * avg,
                "leg fz = {} should be ≈ avg = {avg}",
                f.z
            );
        }
    }

    /// Friction cone: lateral force per stance foot should never
    /// exceed μ·f_z.
    #[test]
    fn friction_cone_respected() {
        let cfg = SrbdMpcConfig::default();
        let mpc = SrbdMpc::new(cfg.clone());
        let s = hover_state();
        let n = cfg.horizon_steps;
        let reference = ReferenceTrajectory::constant(s, n);
        let contact = ContactSchedule::all_stance(n);
        let feet = FootOffsets::constant_per_leg(nominal_feet(), n);
        let sol = mpc.solve(s, &reference, &contact, &feet);
        assert!(sol.solved);
        for (leg_idx, f) in sol.grfs_first_step.iter().enumerate() {
            let bound = cfg.friction_mu * f.z;
            // Allow tiny numerical slack (clarabel converges to ε).
            assert!(
                f.x.abs() <= bound + 1e-3,
                "leg {leg_idx} fx = {} > μ fz = {bound}",
                f.x
            );
            assert!(
                f.y.abs() <= bound + 1e-3,
                "leg {leg_idx} fy = {} > μ fz = {bound}",
                f.y
            );
        }
    }

    /// Step B: `enable_foot_offset` adds 12 Δr columns to the QP.
    /// With a deliberate angular-state error and zero command, the
    /// MPC should still solve and produce a foot-offset solution
    /// within the configured workspace bound.
    #[test]
    fn foot_offset_extension_solves_and_respects_bounds() {
        let mut cfg = SrbdMpcConfig::default();
        cfg.enable_foot_offset = true;
        cfg.max_foot_offset_m = 0.05;
        let mpc = SrbdMpc::new(cfg.clone());

        // Initial state has angular velocity error (body rolling +x).
        let s_now = SrbdState {
            angular_velocity: Vector3::new(0.5, 0.0, 0.0),
            position: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        // Reference holds at hover (no rotation, no translation).
        let s_ref = SrbdState {
            position: Vector3::new(0.0, 0.0, 0.30),
            ..Default::default()
        };
        let n = cfg.horizon_steps;
        let reference = ReferenceTrajectory::constant(s_ref, n);
        let contact = ContactSchedule::all_stance(n);
        let feet = FootOffsets::constant_per_leg(nominal_feet(), n);

        let sol = mpc.solve(s_now, &reference, &contact, &feet);
        assert!(sol.solved, "QP failed to solve with foot-offset extension");
        // Δr bounds: |Δr_ij| ≤ 0.05 m for every leg, every axis, every step.
        for (k, step_offsets) in sol.foot_offsets_all_steps.iter().enumerate() {
            for (leg_idx, dr) in step_offsets.iter().enumerate() {
                for axis in 0..3 {
                    assert!(
                        dr[axis].abs() <= cfg.max_foot_offset_m + 1e-3,
                        "step {k} leg {leg_idx} axis {axis}: |Δr| = {} > bound {}",
                        dr[axis].abs(),
                        cfg.max_foot_offset_m,
                    );
                }
            }
        }
    }

    /// Step B: when `enable_foot_offset = false` (default), the
    /// `foot_offsets_*` fields stay at zero so downstream callers can
    /// safely add them to the Raibert step without altering legacy
    /// behaviour.
    #[test]
    fn foot_offset_disabled_yields_zeros() {
        let cfg = SrbdMpcConfig::default();
        assert!(!cfg.enable_foot_offset, "default must be opt-in");
        let mpc = SrbdMpc::new(cfg.clone());
        let s = hover_state();
        let n = cfg.horizon_steps;
        let reference = ReferenceTrajectory::constant(s, n);
        let contact = ContactSchedule::all_stance(n);
        let feet = FootOffsets::constant_per_leg(nominal_feet(), n);
        let sol = mpc.solve(s, &reference, &contact, &feet);
        assert!(sol.solved);
        for dr in sol.foot_offsets_first_step.iter() {
            assert_eq!(dr.norm(), 0.0, "default config must leave offsets at zero");
        }
    }

    /// A swing leg (no contact for the whole horizon) must produce
    /// zero force (the equality constraint).
    #[test]
    fn swing_leg_produces_zero_force() {
        let cfg = SrbdMpcConfig::default();
        let mpc = SrbdMpc::new(cfg.clone());
        let s = hover_state();
        let n = cfg.horizon_steps;
        let reference = ReferenceTrajectory::constant(s, n);
        // Mark FL as swing for the whole horizon.
        let mut contact = ContactSchedule::all_stance(n);
        contact.is_stance[0] = vec![false; n];
        let feet = FootOffsets::constant_per_leg(nominal_feet(), n);
        let sol = mpc.solve(s, &reference, &contact, &feet);
        assert!(sol.solved);
        let f_fl = sol.grfs_first_step[0];
        assert!(
            f_fl.norm() < 1e-3,
            "FL is in swing, force must be zero, got {f_fl}",
        );
        // Other legs still bear weight.
        let total_other: f64 = sol.grfs_first_step[1..].iter().map(|f| f.z).sum();
        let weight = cfg.mass_kg * 9.81;
        assert!(
            (total_other - weight).abs() < 0.20 * weight,
            "Σf_z (other legs) = {total_other} should still ≈ weight",
        );
    }
}
