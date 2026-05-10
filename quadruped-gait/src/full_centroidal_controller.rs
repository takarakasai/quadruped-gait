//! 24-state full-centroidal gait controller (D3.3.5).
//!
//! Architectural sibling of [`crate::CentroidalMpcGaitController`]:
//! identical open-loop layer (CHAMP-style phase + Raibert footstep +
//! analytical 3R IK → 12 joint targets), but swaps in
//! [`crate::FullCentroidalMpc`] for the closed-loop GRF + joint-velocity
//! prediction.
//!
//! What changes vs. the 12-state centroidal controller:
//!
//! 1. The MPC state now carries the 12 leg joint angles, so the
//!    per-node moment arm `r = R · (foot_body(q) − com_offset)` updates
//!    when the optimiser perturbs `joint_q` within the horizon. The
//!    body-root SRBD and the centroidal-SRBD couldn't see this coupling.
//! 2. The MPC equality constraint set includes **stance no-slip**
//!    (`v_foot_world = 0` per stance-leg-step), expressed linearly in
//!    the condensed QP via the lifted state. This forces the solution
//!    `joint_v` to be physically consistent with a pinned foot.
//! 3. Reference joint_q is held at the controller's current IK output
//!    (D3.3.5a — design choice (a) from the planning session). Swing
//!    leg foot tracking is still driven by the CHAMP layer's joint
//!    target. A future revision can replace the held reference with a
//!    per-step IK of the planned footstep trajectory.
//!
//! GRF output is projected into [`MpcSolution`] via
//! [`to_compat_mpc_solution_full`] so WBC integration stays
//! mode-agnostic. The native solution is available via
//! [`Self::predicted_full_centroidal_solution`].

use nalgebra::Vector3;

use crate::body_state::BodyState;
use crate::config::{GaitConfig, KinematicsConfig, LegId, LegKinematics, VelocityCmd};
use crate::controller::{ControllerOutput, LegOutput};
use crate::footstep::Footstep;
use crate::full_centroidal_mpc::{
    FullCentroidalContactSchedule, FullCentroidalInput, FullCentroidalMpc,
    FullCentroidalMpcConfig, FullCentroidalMpcSolution, FullCentroidalReference,
    FullCentroidalState, N_FEET, N_LEG_JOINTS,
};
use crate::ik::{foot_jacobian_body, solve_leg_ik, LegIkSolution};
use crate::mpc_controller::{
    body_to_world_horizontal, effective_swing_height, make_leg_output,
    world_to_body_horizontal, DEFAULT_CAPTURE_POINT_GAIN_S, HORIZON_STEPS,
    MIN_HALF_FRACTION, STANCE_GRF_MIN_N,
};
use crate::phase::PhaseGenerator;
use crate::srbd_mpc::{MpcSolution, SrbdState};
use crate::swing_traj::swing_position;

#[derive(Clone, Debug)]
pub struct FullCentroidalMpcGaitController {
    cfg: GaitConfig,
    kin: KinematicsConfig,
    phase_gen: PhaseGenerator,
    body_state: BodyState,
    cmd: VelocityCmd,
    knee_forward: [bool; 4],

    k_capture: f64,
    v_observed_world: Vector3<f64>,
    omega_observed_world: Vector3<f64>,

    full_centroidal_mpc: FullCentroidalMpc,
    last_solution: Option<FullCentroidalMpcSolution>,
    last_solution_compat: Option<MpcSolution>,
    mpc_solve_accumulator_s: f64,
}

impl FullCentroidalMpcGaitController {
    pub fn new(cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        let phase_gen = PhaseGenerator::new(cfg.clone());
        // Default config uses a placeholder KinematicsConfig (Cheetah-3
        // class). The host's auto_detect overrides it via
        // `set_full_centroidal_mpc_config` at `GaitController::build`
        // time, slotting in the per-leg analytical FK params for this
        // specific URDF.
        let mut mpc_cfg = FullCentroidalMpcConfig::default_with_kin(kin.clone());
        let _ = &mut mpc_cfg;
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
            full_centroidal_mpc: FullCentroidalMpc::new(mpc_cfg),
            last_solution: None,
            last_solution_compat: None,
            mpc_solve_accumulator_s: f64::INFINITY,
        }
    }

    pub fn predicted_grfs(&self) -> Option<&MpcSolution> {
        self.last_solution_compat.as_ref()
    }

    pub fn predicted_full_centroidal_solution(
        &self,
    ) -> Option<&FullCentroidalMpcSolution> {
        self.last_solution.as_ref()
    }

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
        for slot in 0..N_FEET {
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

    pub fn set_full_centroidal_mpc_config(&mut self, cfg: FullCentroidalMpcConfig) {
        self.full_centroidal_mpc.set_config(cfg);
    }
    pub fn full_centroidal_mpc_config(&self) -> &FullCentroidalMpcConfig {
        self.full_centroidal_mpc.config()
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
        // The MPC config carries its own copy of `kin` for FK; keep them
        // in sync when the host re-tunes the kinematics.
        let mut mpc_cfg = self.full_centroidal_mpc.config().clone();
        mpc_cfg.kinematics = kin.clone();
        self.full_centroidal_mpc.set_config(mpc_cfg);
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

        let dt_per_step = self.full_centroidal_mpc.config().dt_per_step;
        self.mpc_solve_accumulator_s += dt;
        if self.mpc_solve_accumulator_s >= dt_per_step {
            let sol = self.solve_full_centroidal_mpc(&output);
            self.last_solution_compat = Some(to_compat_mpc_solution_full(&sol));
            self.last_solution = Some(sol);
            self.mpc_solve_accumulator_s = 0.0;
        }

        output
    }

    fn solve_full_centroidal_mpc(
        &mut self,
        output: &ControllerOutput,
    ) -> FullCentroidalMpcSolution {
        let cfg = self.full_centroidal_mpc.config().clone();
        let n = cfg.horizon_steps;

        // Current joint_q from the IK output (12 entries, FL/FR/RL/RR ×
        // [hip, thigh, calf]). These feed the per-node FK in the MPC so
        // the moment arm at step 0 matches what the legs are actually
        // doing.
        let mut joint_q_now = [0.0_f64; N_LEG_JOINTS];
        for slot in 0..N_FEET {
            let leg = &output.legs[slot];
            joint_q_now[3 * slot] = leg.q_hip;
            joint_q_now[3 * slot + 1] = leg.q_thigh;
            joint_q_now[3 * slot + 2] = leg.q_calf;
        }

        let s_now = FullCentroidalState {
            v_com_world: self.v_observed_world,
            angular_velocity_world: self.omega_observed_world,
            base_pos_world: self.body_state.world_position,
            base_euler_zyx: Vector3::new(0.0, 0.0, self.body_state.world_yaw),
            joint_q: joint_q_now,
        };

        // Build reference: cmd-velocity integrated body trajectory +
        // held joint_q + joint_v=0 + gravity-balanced GRF.
        //
        // joint_q held constant over the horizon (D3.3.5a simplification
        // — design choice (a)). The MPC's stance no-slip constraint will
        // still produce non-zero joint_v as needed to keep stance feet
        // pinned; the cost just doesn't bias swing legs to follow the
        // open-loop footstep trajectory in this revision.
        let v_world_cmd = body_to_world_horizontal(
            Vector3::new(self.cmd.vx, self.cmd.vy, 0.0),
            self.body_state.world_yaw,
        );
        let dt_per_step = cfg.dt_per_step;
        let stance_now: [bool; N_FEET] = [
            output.legs[0].phase.is_stance,
            output.legs[1].phase.is_stance,
            output.legs[2].phase.is_stance,
            output.legs[3].phase.is_stance,
        ];
        let holding = self.cmd.is_zero();

        // Per-step contact schedule (mirrors 12-state simplification:
        // step 0 = actual phase, subsequent steps = duty > 0.5 ?
        // all stance : all swing).
        let mut contact = FullCentroidalContactSchedule {
            is_stance: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        };
        for leg in 0..N_FEET {
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

        // Per-step reference state + input. Body pose integrates the cmd
        // velocity; joint_q held; gravity distributed across stance legs
        // for the GRF reference (the QP deviates as needed for the cost
        // and constraints).
        let mut ref_states = Vec::with_capacity(n);
        let mut ref_inputs = Vec::with_capacity(n);
        for k in 0..n {
            let t = (k + 1) as f64 * dt_per_step;
            let mut sk = s_now;
            sk.v_com_world = v_world_cmd;
            sk.angular_velocity_world = Vector3::new(0.0, 0.0, self.cmd.wz);
            sk.base_pos_world = s_now.base_pos_world + v_world_cmd * t;
            sk.base_euler_zyx.z = s_now.base_euler_zyx.z + self.cmd.wz * t;
            ref_states.push(sk);

            // Gravity-balanced GRF reference: total = m·g, split across
            // legs in stance at this step. Swing legs get 0.
            let n_stance = (0..N_FEET).filter(|&l| contact.is_stance[l][k]).count();
            let f_per_stance = if n_stance > 0 {
                cfg.mass_kg * 9.81 / n_stance as f64
            } else {
                0.0
            };
            let mut grfs = [Vector3::zeros(); N_FEET];
            for leg in 0..N_FEET {
                if contact.is_stance[leg][k] {
                    grfs[leg].z = f_per_stance;
                }
            }
            ref_inputs.push(FullCentroidalInput {
                grfs_world: grfs,
                joint_v: [0.0; N_LEG_JOINTS],
            });
        }
        let reference = FullCentroidalReference {
            states: ref_states,
            inputs: ref_inputs,
        };

        self.full_centroidal_mpc.solve(s_now, &reference, &contact)
    }

    /// Footstep planner — identical to the 12-state version. Duplicated
    /// (not delegated) so the two controllers can be evaluated head-to-
    /// head without state leak.
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

/// Lossy projection of `FullCentroidalMpcSolution` into the SRBD-shaped
/// `MpcSolution` for WBC integration compat. Same idea as
/// `to_compat_mpc_solution` in `centroidal_controller.rs`.
fn to_compat_mpc_solution_full(sol: &FullCentroidalMpcSolution) -> MpcSolution {
    let predicted_body_states: Vec<SrbdState> = sol
        .predicted_states
        .iter()
        .map(|s| SrbdState {
            orientation_rpy: s.base_euler_zyx,
            position: s.base_pos_world,
            angular_velocity: s.angular_velocity_world,
            linear_velocity: s.v_com_world,
        })
        .collect();
    let grfs_first_step = sol.first_input.grfs_world;
    let grfs_all_steps: Vec<[Vector3<f64>; N_FEET]> = sol
        .inputs_all_steps
        .iter()
        .map(|u| u.grfs_world)
        .collect();
    MpcSolution {
        grfs_first_step,
        grfs_all_steps,
        predicted_body_states,
        objective: sol.objective,
        solved: sol.solved,
    }
}
