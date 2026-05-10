//! Centroidal-SRBD gait controller.
//!
//! Architectural sibling of [`crate::MpcGaitController`]: same input
//! shape (velocity command + per-tick `dt`), same output shape
//! ([`ControllerOutput`] with 12 joint targets), same footstep
//! planner (Raibert + capture-point + LIP horizon look-ahead). What's
//! different is the MPC layer:
//!
//! - [`crate::MpcGaitController`] uses the body-root SRBD MPC
//!   ([`crate::SrbdMpc`]).
//! - This controller uses the centroidal-SRBD MPC
//!   ([`crate::CentroidalMpc`]) — same convex QP family, but the
//!   state is expressed in centroidal momentum coordinates so the
//!   robot's CoM offset (e.g. namiashi's +5 mm trunk_inertia shift)
//!   is modelled correctly. See `centroidal_mpc.rs` for the full
//!   motivation.
//!
//! The output GRFs are returned via the same
//! [`crate::srbd_mpc::MpcSolution`] type as `MpcGaitController` so
//! the host's WBC integration can stay mode-agnostic. A second
//! accessor [`Self::predicted_centroidal_solution`] exposes the
//! native centroidal solution for hosts that want the higher-fidelity
//! centroidal-aware path.

use nalgebra::Vector3;

use crate::body_state::BodyState;
use crate::centroidal_mpc::{
    CentroidalContactSchedule, CentroidalFootOffsets, CentroidalMpc, CentroidalMpcConfig,
    CentroidalMpcSolution, CentroidalReference, CentroidalState,
};
use crate::config::{GaitConfig, KinematicsConfig, LegId, LegKinematics, VelocityCmd};
use crate::controller::{ControllerOutput, LegOutput};
use crate::footstep::Footstep;
use crate::ik::{foot_jacobian_body, solve_leg_ik, LegIkSolution};
use crate::mpc_controller::{
    body_to_world_horizontal, effective_swing_height, make_leg_output,
    world_to_body_horizontal, DEFAULT_CAPTURE_POINT_GAIN_S, HORIZON_STEPS,
    MIN_HALF_FRACTION, STANCE_GRF_MIN_N,
};
use crate::phase::PhaseGenerator;
use crate::srbd_mpc::{MpcSolution, SrbdState};
use crate::swing_traj::swing_position;

/// Centroidal-SRBD gait controller. See module-level docs.
#[derive(Clone, Debug)]
pub struct CentroidalMpcGaitController {
    cfg: GaitConfig,
    kin: KinematicsConfig,
    phase_gen: PhaseGenerator,
    body_state: BodyState,
    cmd: VelocityCmd,
    knee_forward: [bool; 4],

    /// Capture-point feedback gain in seconds (`√(h/g)` of the LIP
    /// model). Same semantics as `MpcGaitController`.
    k_capture: f64,
    v_observed_world: Vector3<f64>,
    omega_observed_world: Vector3<f64>,

    /// Centroidal-SRBD convex MPC. Re-solved at most once per
    /// `dt_per_step` window (default 30 ms).
    centroidal_mpc: CentroidalMpc,
    /// Latest native centroidal solution. Held between solves.
    last_solution: Option<CentroidalMpcSolution>,
    /// SRBD-shaped projection of `last_solution` for compat with
    /// hosts that consume `MpcSolution`. Kept in sync with
    /// `last_solution` after each solve.
    last_solution_compat: Option<MpcSolution>,
    mpc_solve_accumulator_s: f64,
}

impl CentroidalMpcGaitController {
    pub fn new(cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        let phase_gen = PhaseGenerator::new(cfg.clone());
        Self {
            cfg,
            kin,
            phase_gen,
            body_state: BodyState::new(),
            cmd: VelocityCmd::zero(),
            knee_forward: [false; 4],
            k_capture: DEFAULT_CAPTURE_POINT_GAIN_S,
            v_observed_world: Vector3::zeros(),
            omega_observed_world: Vector3::zeros(),
            centroidal_mpc: CentroidalMpc::new(CentroidalMpcConfig::default()),
            last_solution: None,
            last_solution_compat: None,
            mpc_solve_accumulator_s: f64::INFINITY,
        }
    }

    /// Predicted GRFs as [`MpcSolution`] (SRBD-shaped) for WBC compat.
    /// The numeric GRF values are the centroidal MPC's solution; only
    /// the wrapper struct shape matches SRBD's. `predicted_body_states`
    /// is a lossy SRBD projection of the centroidal state.
    pub fn predicted_grfs(&self) -> Option<&MpcSolution> {
        self.last_solution_compat.as_ref()
    }

    /// Native centroidal MPC solution (no compat projection). Hosts
    /// that want the centroidal-aware WBC integration read this.
    pub fn predicted_centroidal_solution(&self) -> Option<&CentroidalMpcSolution> {
        self.last_solution.as_ref()
    }

    /// Per-leg `τ = -J^T·f_GRF` torque feedforward. Identical to
    /// [`crate::MpcGaitController::stance_grf_torques`] but reads from
    /// the centroidal-MPC solution.
    pub fn stance_grf_torques(
        &self,
        output: &ControllerOutput,
    ) -> [Option<[f64; 3]>; 4] {
        let mut out = [None; 4];
        let Some(sol) = self.last_solution_compat.as_ref() else {
            return out;
        };
        if !sol.solved {
            return out;
        }
        let yaw = self.body_state.world_yaw;
        let (sy, cy) = yaw.sin_cos();
        for slot in 0..4 {
            let leg_out = &output.legs[slot];
            if !leg_out.phase.is_stance {
                continue;
            }
            let f_world = sol.grfs_first_step[slot];
            if f_world.norm() < STANCE_GRF_MIN_N {
                continue;
            }
            let f_body = Vector3::new(
                cy * f_world.x + sy * f_world.y,
                -sy * f_world.x + cy * f_world.y,
                f_world.z,
            );
            let kin_leg = self.kin.leg(LegId::ALL[slot]);
            let j = foot_jacobian_body(
                kin_leg,
                leg_out.q_hip,
                leg_out.q_thigh,
                leg_out.q_calf,
            );
            let tau = -(j.transpose() * f_body);
            out[slot] = Some([tau.x, tau.y, tau.z]);
        }
        out
    }

    pub fn set_centroidal_mpc_config(&mut self, cfg: CentroidalMpcConfig) {
        self.centroidal_mpc.set_config(cfg);
    }
    pub fn centroidal_mpc_config(&self) -> &CentroidalMpcConfig {
        self.centroidal_mpc.config()
    }

    pub fn body_state(&self) -> &BodyState {
        &self.body_state
    }

    pub fn velocity_cmd(&self) -> VelocityCmd {
        self.cmd
    }
    pub fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        self.cmd = cmd;
    }

    pub fn config(&self) -> &GaitConfig {
        &self.cfg
    }
    pub fn set_config(&mut self, cfg: GaitConfig) {
        self.cfg = cfg.clone();
        self.phase_gen = PhaseGenerator::new(cfg);
    }

    pub fn kinematics(&self) -> &KinematicsConfig {
        &self.kin
    }
    pub fn set_kinematics(&mut self, kin: KinematicsConfig) {
        self.kin = kin;
    }

    pub fn knee_forward(&self) -> [bool; 4] {
        self.knee_forward
    }
    pub fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        self.knee_forward[crate::controller::slot_of(leg)] = forward;
    }
    pub fn set_knee_pattern(&mut self, pattern: crate::config::KneePattern) {
        self.knee_forward = pattern.to_knee_forward();
    }
    pub fn knee_pattern(&self) -> crate::config::KneePattern {
        crate::config::KneePattern::from_knee_forward(self.knee_forward)
    }

    pub fn capture_point_gain(&self) -> f64 {
        self.k_capture
    }
    pub fn set_capture_point_gain(&mut self, k: f64) {
        self.k_capture = k.max(0.0);
    }

    pub fn set_body_state_observed(
        &mut self,
        v_world: Vector3<f64>,
        omega_world: Vector3<f64>,
    ) {
        self.v_observed_world = v_world;
        self.omega_observed_world = omega_world;
    }

    pub fn set_body_pose_observed(
        &mut self,
        world_yaw: f64,
        world_position: Vector3<f64>,
    ) {
        self.body_state.world_yaw = world_yaw;
        self.body_state.world_position = world_position;
    }

    pub fn reset(&mut self) {
        self.body_state = BodyState::new();
        self.phase_gen.reset();
        self.cmd = VelocityCmd::zero();
        self.last_solution = None;
        self.last_solution_compat = None;
        self.mpc_solve_accumulator_s = f64::INFINITY;
    }

    /// One control tick: advance phase, update body integrator, plan
    /// footsteps, IK to joint angles. Re-solves the centroidal MPC at
    /// most once per `dt_per_step` window.
    pub fn tick(&mut self, dt: f64) -> ControllerOutput {
        self.phase_gen.advance(dt, &self.cmd);
        self.body_state.integrate(&self.cmd, dt);

        let v_obs_body = world_to_body_horizontal(
            self.v_observed_world,
            self.body_state.world_yaw,
        );
        let v_cmd = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
        let v_err_body = v_obs_body - v_cmd;

        let phases = self.phase_gen.legs();
        let mut legs: [Option<LegOutput>; 4] = [None, None, None, None];
        for ps in phases.iter() {
            let kin_leg = self.kin.leg(ps.leg);
            let footstep = self.compute_mpc_footstep(kin_leg, &v_err_body);
            let target = if ps.is_stance {
                footstep.stance_at(ps.sub_fraction)
            } else {
                let swing_h = effective_swing_height(self.cfg.swing_height_m, &self.cmd);
                swing_position(
                    footstep.lift_off,
                    footstep.touch_down,
                    swing_h,
                    ps.sub_fraction,
                )
            };
            let knee_fwd = self.knee_forward[crate::controller::slot_of(ps.leg)];
            let sol = solve_leg_ik(kin_leg, target, knee_fwd);
            let reachable = matches!(sol, LegIkSolution::Reached { .. });
            let (h, t, c) = sol.angles();
            legs[crate::controller::slot_of(ps.leg)] = Some(make_leg_output(
                ps.leg, kin_leg, *ps, footstep, target, h, t, c, reachable,
            ));
        }
        let output = ControllerOutput {
            legs: legs.map(|x| x.expect("all four legs filled by phase loop")),
            body_state: self.body_state,
        };

        let dt_per_step = self.centroidal_mpc.config().dt_per_step;
        self.mpc_solve_accumulator_s += dt;
        if self.mpc_solve_accumulator_s >= dt_per_step {
            let centroidal_sol = self.solve_centroidal_mpc(&output);
            self.last_solution_compat = Some(to_compat_mpc_solution(&centroidal_sol));
            self.last_solution = Some(centroidal_sol);
            self.mpc_solve_accumulator_s = 0.0;
        }

        output
    }

    /// Build the centroidal MPC inputs from current state and call
    /// the QP solver. Mirror of [`crate::MpcGaitController::solve_srbd_mpc`].
    fn solve_centroidal_mpc(&self, output: &ControllerOutput) -> CentroidalMpcSolution {
        let cfg = self.centroidal_mpc.config().clone();
        let n = cfg.horizon_steps;

        // Current centroidal state. h_lin_per_mass = observed CoM
        // velocity ≈ observed body-root velocity (the difference is
        // ω × com_offset, O(mm/s) at typical gait speeds — well below
        // MPC's noise floor). angular_velocity_world is the observed
        // gyro directly (D1.4 — was h_ang/m which forced unit
        // gymnastics in the cost matrix).
        let s_now = CentroidalState {
            h_lin_per_mass: self.v_observed_world,
            angular_velocity_world: self.omega_observed_world,
            base_pos_world: self.body_state.world_position,
            base_euler_zyx: Vector3::new(0.0, 0.0, self.body_state.world_yaw),
        };

        // Reference: track commanded velocity, integrate position + yaw.
        let v_world_cmd = body_to_world_horizontal(
            Vector3::new(self.cmd.vx, self.cmd.vy, 0.0),
            self.body_state.world_yaw,
        );
        let reference = CentroidalReference::from_constant_velocity(
            s_now,
            v_world_cmd,
            self.cmd.wz,
            &cfg,
        );

        // Contact schedule: same hold-mode special-case as the SRBD MPC
        // — at zero cmd, all 4 legs in stance for every horizon step
        // (otherwise the MPC sees collapsing-support and fires huge
        // step-0 impulses).
        let holding = self.cmd.is_zero();
        let stance_now: [bool; 4] = [
            output.legs[0].phase.is_stance,
            output.legs[1].phase.is_stance,
            output.legs[2].phase.is_stance,
            output.legs[3].phase.is_stance,
        ];
        let mut contact = CentroidalContactSchedule {
            is_stance: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        };
        for leg in 0..4 {
            for k in 0..n {
                let in_stance = if holding {
                    true
                } else if k == 0 {
                    stance_now[leg]
                } else {
                    self.cfg.duty_factor > 0.5
                };
                contact.is_stance[leg].push(in_stance);
            }
        }

        // Foot offsets: r_i = foot_world − CoM_world. CoM_world is
        // body_pos + R · com_offset_body. For yaw-only and small
        // com_offset, this is foot_body_rotated_to_world − com_offset_world.
        let yaw = self.body_state.world_yaw;
        let com_offset_world =
            body_to_world_horizontal(cfg.com_offset_body, yaw);
        let foot_rel_com_world: [Vector3<f64>; 4] = [
            body_to_world_horizontal(output.legs[0].foot_body, yaw) - com_offset_world,
            body_to_world_horizontal(output.legs[1].foot_body, yaw) - com_offset_world,
            body_to_world_horizontal(output.legs[2].foot_body, yaw) - com_offset_world,
            body_to_world_horizontal(output.legs[3].foot_body, yaw) - com_offset_world,
        ];
        let feet = CentroidalFootOffsets::constant_per_leg(foot_rel_com_world, n);

        self.centroidal_mpc.solve(s_now, &reference, &contact, &feet)
    }

    /// Footstep planner identical to
    /// [`crate::MpcGaitController::compute_mpc_footstep`]. Duplicated
    /// here so the two controllers can be evaluated head-to-head
    /// without one's state leaking into the other's solver.
    fn compute_mpc_footstep(
        &self,
        kin: &LegKinematics,
        v_err_body: &Vector3<f64>,
    ) -> Footstep {
        let stance_duration = self.cfg.cycle_period_s * self.cfg.duty_factor;
        let v_body = Vector3::new(self.cmd.vx, self.cmd.vy, 0.0);
        let omega = Vector3::new(0.0, 0.0, self.cmd.wz);
        let v_hip = v_body + omega.cross(&kin.hip_offset);
        let mut half = v_hip * (0.5 * stance_duration);

        let feedback_enabled = !self.cmd.is_zero();
        let mut feedback = Vector3::zeros();
        if feedback_enabled {
            feedback.x = self.k_capture * v_err_body.x;
            feedback.y = self.k_capture * v_err_body.y;
        }
        let horizon_weight = 1.0 / HORIZON_STEPS as f64;
        let mut horizon_bias = Vector3::zeros();
        if feedback_enabled {
            horizon_bias.x = horizon_weight * self.k_capture * v_err_body.x;
            horizon_bias.y = horizon_weight * self.k_capture * v_err_body.y;
        }

        let closed_loop = feedback + horizon_bias;
        let raw_half = half + closed_loop;
        let mut combined = raw_half;
        let min_x = MIN_HALF_FRACTION * half.x;
        let min_y = MIN_HALF_FRACTION * half.y;
        if half.x > 0.0 && combined.x < min_x {
            combined.x = min_x;
        } else if half.x < 0.0 && combined.x > min_x {
            combined.x = min_x;
        }
        if half.y > 0.0 && combined.y < min_y {
            combined.y = min_y;
        } else if half.y < 0.0 && combined.y > min_y {
            combined.y = min_y;
        }
        half = combined;
        let max_half = 0.5 * self.cfg.max_step_length_m;
        let mag = half.norm();
        if mag > max_half && mag > 0.0 {
            half *= max_half / mag;
        }
        Footstep {
            lift_off: kin.nominal_foot_body - half,
            touch_down: kin.nominal_foot_body + half,
        }
    }
}

/// Lossy projection of [`CentroidalMpcSolution`] into the SRBD-shaped
/// [`MpcSolution`]. Field semantics differ:
/// - GRFs: identical (both world-frame N).
/// - `predicted_body_states`: SRBD's `[orientation_rpy; position; ω; v]`
///   are filled from the centroidal state's `[euler; pos; ω; v_com]`
///   directly (D1.4 — no unit conversion needed now that the
///   centroidal state stores ω). For visualisation / diagnostic only —
///   host code that needs the centroidal state should call
///   `predicted_centroidal_solution` instead.
fn to_compat_mpc_solution(sol: &CentroidalMpcSolution) -> MpcSolution {
    let predicted_body_states: Vec<SrbdState> = sol
        .predicted_body_states
        .iter()
        .map(|s| SrbdState {
            orientation_rpy: s.base_euler_zyx,
            position: s.base_pos_world,
            angular_velocity: s.angular_velocity_world,
            linear_velocity: s.h_lin_per_mass,
        })
        .collect();
    MpcSolution {
        grfs_first_step: sol.grfs_first_step,
        grfs_all_steps: sol.grfs_all_steps.clone(),
        predicted_body_states,
        objective: sol.objective,
        solved: sol.solved,
    }
}
