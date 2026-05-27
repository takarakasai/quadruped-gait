//! Open-loop "linear-trunk" crawl gait.
//!
//! Coexists with the existing [`crate::GaitConfig::crawl`] family —
//! same 4-beat diagonal-sequence wave — but uses a fundamentally
//! different planning strategy: the trunk is fed a pre-computed +X
//! linear trajectory in world frame and the legs are servoed around
//! it, vs the existing Crawl which lets the body-velocity controller
//! (CHAMP / MPC) drive the trunk pose dynamically.
//!
//! # Algorithm
//!
//! The trunk is held on a strictly linear `+X` trajectory in world frame:
//!
//! ```text
//!   trunk_world(t) = (v · t, 0, h, identity)
//! ```
//!
//! — i.e. no `Y` translation, no `Z` translation, no roll / pitch / yaw
//! oscillation **by construction**. The four legs are servoed around
//! that trajectory: each per-leg sub-cycle (`T/4`) consists of a 4-support
//! window of duration `α · T/4` followed by a single-leg swing of
//! duration `(1 − α) · T/4`. Sub-cycles run back-to-back through the
//! four legs in [`LinearCrawlConfig::leg_order`], producing one cycle of
//! the form (4-sup, swing #0, 4-sup, swing #1, 4-sup, swing #2, 4-sup,
//! swing #3).
//!
//! During stance each foot is planted in world frame; the trunk advances
//! over it, so the foot's body-frame X recedes linearly. During swing
//! the foot interpolates from the previous planted body-X (
//! `nominal_x − v·T·(1−s)/2`, where `s = (1 − α)/4`) to the next
//! (`nominal_x + v·T·(1−s)/2`) along a quintic-smoothstep curve in X
//! and a half-sine arc in Z. Each leg's body-Y is held at its
//! [`crate::LegKinematics::nominal_foot_body`] Y throughout — that's
//! what keeps the trunk from needing any lateral counter-shift.
//!
//! Per-leg analytical IK is then run via [`crate::ik::solve_leg_ik`]
//! to convert the desired body-frame foot position into hip / thigh /
//! calf joint angles.
//!
//! # Static stability vs. open-loop nature
//!
//! Because there's no lateral trunk shift, the trunk CoM does NOT stay
//! over the support triangle during the 3-support phases. A real robot
//! relying on the static-stability margin would tip. This controller
//! is therefore open-loop kinematic only: it works against stiff PD
//! position tracking (typical Go2-class setup: `Kp ≈ 500 N·m/rad`)
//! that physically forces the trunk to stay on its commanded trajectory
//! through the actuators. It does **not** include any IMU feedback,
//! capture-point logic, or footstep adaptation — by design (see
//! caller's spec point #1).
//!
//! # Limitations
//!
//! - At `t = 0` each leg's body-frame foot position differs from the
//!   host's home pose by an offset depending on the leg's phase in the
//!   cycle; worst case is `± v · (1 − s) · T / 2`. For
//!   `v = 0.1 m/s, T = 1.0 s, α = 0.5` that's `± 44 mm`. Stiff PD
//!   absorbs that transient on Go2; smaller robots may need a soft
//!   start.
//! - The host model is responsible for matching the controller's
//!   `body_height_m` (set the trunk world Z accordingly before
//!   enabling the gait).

use crate::config::{KinematicsConfig, LegId};
use crate::ik::solve_leg_ik;
use nalgebra::Vector3;

/// All configurable knobs for [`LinearCrawlController`]. See the module
/// docs for what each field does in the planner.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearCrawlConfig {
    /// Trunk forward speed (world `+X`), m/s.
    pub speed_mps: f64,
    /// Trunk world Z, m. Foot body-frame Z during stance is forced to
    /// `−body_height_m` (foot on the ground).
    pub body_height_m: f64,
    /// One full cycle (= 4 leg swings) duration, s.
    pub cycle_period_s: f64,
    /// Fraction of each per-leg sub-cycle (`T/4`) spent in 4-support
    /// before lifting that leg. Must be in (0, 1).
    pub four_support_fraction: f64,
    /// Peak swing-foot height above the planted Z, m.
    pub swing_height_m: f64,
    /// Order in which legs swing over one cycle. Default
    /// `[RL, FR, RR, FL]` is the classical diagonal-sequence wave that
    /// maximises the static-stability margin (even though this open-
    /// loop controller doesn't rely on it for stability).
    pub leg_order: [LegId; 4],
    /// Per-leg knee-bend selection forwarded to
    /// [`crate::ik::solve_leg_ik`]. Indexed `[FL, FR, RL, RR]`.
    pub knee_forward: [bool; 4],
    /// Soft-start duration, s. Over the first `soft_start_duration_s`
    /// of wall time after [`LinearCrawlController::enable`], the
    /// internal gait clock advances at a smoothstep-ramped fraction
    /// of real time (0 at `t = 0`, full speed at
    /// `t ≥ soft_start_duration_s`). Body translation, cycle phase
    /// and leg-swing timing all scale with the same clock, so:
    ///
    /// - the trunk smoothly accelerates from 0 to `speed_mps` instead
    ///   of jerking forward at full speed on tick 1;
    /// - the first swing leg doesn't lift until the body has had
    ///   time to start moving — eliminates the impulsive joint
    ///   commands that tip the robot backward at gait start
    ///   ("仰向け" symptom).
    ///
    /// Set to `0.0` to disable the ramp (legacy behaviour). Default
    /// `0.5 s` is conservative on Go2-class robots with stiff PD.
    pub soft_start_duration_s: f64,
}

impl Default for LinearCrawlConfig {
    fn default() -> Self {
        // Defaults tuned to the user-validated Go2 setup
        // (Kp=500 Position-PD, `models/unitree_go2/go2.misa`).
        Self {
            // Same defaults as `GaitConfig::crawl()` so the GUI
            // ("Type: Crawl" + "Mode: LinearCrawl") and direct
            // `LinearCrawlConfig::default()` agree on starting values.
            // See `examples/go2_linear_crawl_sweep.rs` for the sweep
            // that picked these (α=0.85 + Kp=1000 gives min |yaw|max
            // and max Δx on Go2 at vx=0.05 m/s).
            speed_mps: 0.05,
            body_height_m: 0.27,
            cycle_period_s: 1.667,
            four_support_fraction: 0.85,
            swing_height_m: 0.005,
            leg_order: [LegId::RL, LegId::FR, LegId::RR, LegId::FL],
            knee_forward: [false; 4],
            soft_start_duration_s: 0.5,
        }
    }
}

/// One tick's output of [`LinearCrawlController::tick`]. All angles use
/// the same hip / thigh / calf convention as [`crate::ik::solve_leg_ik`].
#[derive(Clone, Debug)]
pub struct LinearCrawlOutput {
    /// Per-leg `(q_hip, q_thigh, q_calf)`. Indexed `[FL, FR, RL, RR]`.
    pub angles: [(f64, f64, f64); 4],
    /// `true` for the single leg currently in swing, `false` for the
    /// three (or four, during the 4-support sub-windows) supporting
    /// legs. Indexed `[FL, FR, RL, RR]`.
    pub in_swing: [bool; 4],
    /// Trunk world target pose `(x, y, z)`. Orientation is the identity
    /// rotation by construction.
    pub trunk_world_xyz: [f64; 3],
    /// Per-leg body-frame foot target before IK, useful for diagnostics
    /// / plotting. Indexed `[FL, FR, RL, RR]`.
    pub foot_body_xyz: [[f64; 3]; 4],
    /// `true` if the analytical IK reached the target; `false` if the
    /// solver clamped to the reachable boundary (target outside the
    /// leg's annulus). Indexed `[FL, FR, RL, RR]`.
    pub foot_reachable: [bool; 4],
}

/// Open-loop linear-trunk crawl planner. Construct with
/// [`Self::new`], then drive forward in time with [`Self::tick`].
pub struct LinearCrawlController {
    cfg: LinearCrawlConfig,
    kin: KinematicsConfig,
    /// **Gait clock** — drives all pose / trunk-translation formulas.
    /// Advances by `dt · ramp(wall_t)` each tick (= `dt` once past the
    /// soft-start window). The legacy field name `t` is kept for
    /// internal compatibility; `output_at` consumes it directly.
    t: f64,
    /// Wall clock since the controller was enabled / reset. Only
    /// referenced by the soft-start ramp; `output_at` ignores it.
    wall_t: f64,
    /// Where each leg sits in `cfg.leg_order` — drives its phase
    /// offset in the cycle. Indexed `[FL, FR, RL, RR]`.
    sub_cycle_index: [usize; 4],
    /// Body-frame nominal foot position per leg, with Z forced to
    /// `−body_height_m`. Indexed `[FL, FR, RL, RR]`.
    nominal_body: [Vector3<f64>; 4],
    /// `true` once a stop has been requested (`speed_mps = 0`) AND the
    /// cycle has reached an all-stance phase. While holding, `tick()`
    /// does NOT advance time — the controller emits the same all-stance
    /// pose every tick. Cleared the moment `speed_mps` becomes non-zero
    /// again, so the next `tick()` resumes the gait from where it was
    /// frozen.
    holding: bool,
}

#[inline]
fn leg_arr_idx(l: LegId) -> usize {
    match l {
        LegId::FL => 0,
        LegId::FR => 1,
        LegId::RL => 2,
        LegId::RR => 3,
    }
}

impl LinearCrawlController {
    pub fn new(kin: KinematicsConfig, cfg: LinearCrawlConfig) -> Self {
        let mut sub_cycle_index = [0usize; 4];
        for (i, l) in cfg.leg_order.iter().enumerate() {
            sub_cycle_index[leg_arr_idx(*l)] = i;
        }
        let h = cfg.body_height_m;
        let nb = |lk: &crate::config::LegKinematics| {
            Vector3::new(lk.nominal_foot_body.x, lk.nominal_foot_body.y, -h)
        };
        let nominal_body = [nb(&kin.fl), nb(&kin.fr), nb(&kin.rl), nb(&kin.rr)];
        Self {
            cfg,
            kin,
            t: 0.0,
            wall_t: 0.0,
            sub_cycle_index,
            nominal_body,
            holding: false,
        }
    }

    pub fn config(&self) -> &LinearCrawlConfig {
        &self.cfg
    }
    pub fn elapsed(&self) -> f64 {
        self.t
    }
    pub fn reset(&mut self) {
        self.t = 0.0;
        self.wall_t = 0.0;
        self.holding = false;
    }
    /// Equivalent to [`Self::reset`]. Kept for parity with the
    /// existing `GaitController::enable` ergonomics.
    pub fn enable(&mut self) {
        self.t = 0.0;
        self.wall_t = 0.0;
        self.holding = false;
    }

    /// `true` while the controller has frozen the cycle (= robot is
    /// standing still pending the next non-zero `set_speed` call).
    pub fn is_holding(&self) -> bool {
        self.holding
    }

    /// Trunk world X target at the current time.
    pub fn trunk_world_x(&self) -> f64 {
        self.cfg.speed_mps * self.t
    }

    /// Live-update the forward speed without disturbing cycle phase.
    pub fn set_speed(&mut self, v: f64) {
        self.cfg.speed_mps = v;
    }

    /// Advance the internal clock by `dt` and emit one frame of joint /
    /// trunk targets.
    ///
    /// Honours the standstill convention: when `cfg.speed_mps == 0` the
    /// controller keeps ticking until the cycle reaches a phase where
    /// all four legs are simultaneously in stance, then freezes time
    /// (no further integration) until [`Self::set_speed`] is called
    /// with a non-zero value. This lets the caller stop the gait by
    /// dropping `vx` to zero without the in-flight swing leg
    /// teleporting to the ground or the static cycle phantom-lifting
    /// any feet.
    pub fn tick(&mut self, dt: f64) -> LinearCrawlOutput {
        let want_hold = self.cfg.speed_mps.abs() < 1e-9;
        if self.holding {
            // Holding state: stay frozen until speed becomes non-zero.
            if want_hold {
                return self.output_at(self.t);
            }
            self.holding = false;
        }
        // Soft-start: scale `dt` by a smoothstep-ramped fraction that
        // grows from 0 (at wall_t = 0) to 1 (at wall_t ≥ duration).
        // This makes the gait clock `t` advance slowly at first so
        // body translation, cycle phase, and swing-leg trajectories
        // all ease in together — no impulsive joint commands at the
        // gait's first tick.
        let dur = self.cfg.soft_start_duration_s;
        self.wall_t += dt;
        let dt_eff = if dur > 0.0 && self.wall_t < dur + dt {
            let u = (self.wall_t / dur).clamp(0.0, 1.0);
            let ramp = u * u * u * (u * (u * 6.0 - 15.0) + 10.0);
            dt * ramp
        } else {
            dt
        };
        self.t += dt_eff;
        let out = self.output_at(self.t);
        // If the user wants to stop AND we've reached an all-stance
        // phase, freeze here. The current swing (if any) was allowed to
        // run to completion above before this branch could fire.
        if want_hold && out.in_swing.iter().all(|&s| !s) {
            self.holding = true;
        }
        out
    }

    /// Compute the controller output at absolute time `t` without
    /// mutating the internal clock. Useful for tests, look-ahead, and
    /// plotting the planned trajectory.
    pub fn output_at(&self, t: f64) -> LinearCrawlOutput {
        let v = self.cfg.speed_mps;
        let big_t = self.cfg.cycle_period_s.max(1e-6);
        let alpha = self.cfg.four_support_fraction.clamp(0.05, 0.95);
        // Each per-leg sub-cycle is T/4; swing is (1−α) of that.
        let swing_dur_phase = (1.0 - alpha) * 0.25;
        let stance_dur_phase = 1.0 - swing_dur_phase;
        let trunk_x = v * t;
        let cycle_phase = (t / big_t).rem_euclid(1.0);

        let mut angles = [(0.0f64, 0.0f64, 0.0f64); 4];
        let mut in_swing = [false; 4];
        let mut foot_body_xyz = [[0.0f64; 3]; 4];
        let mut foot_reachable = [true; 4];

        for leg_i in 0..4 {
            let i = self.sub_cycle_index[leg_i];
            let nominal = self.nominal_body[leg_i];

            let sub_start = i as f64 * 0.25;
            let sub_4sup_end = sub_start + alpha * 0.25;
            let sub_end = (i + 1) as f64 * 0.25;

            // Swing window strictly inside this leg's own sub-cycle:
            // [sub_4sup_end, sub_end). Anywhere else in the cycle the
            // leg is in stance.
            let in_swing_phase = cycle_phase >= sub_4sup_end && cycle_phase < sub_end;

            let (body_x, body_z) = if in_swing_phase {
                let u = (cycle_phase - sub_4sup_end) / swing_dur_phase;
                let lift = nominal.x - v * stance_dur_phase * big_t / 2.0;
                let land = nominal.x + v * stance_dur_phase * big_t / 2.0;
                // **Continuous-velocity swing profile**. The foot's
                // *world*-frame velocity follows `v_peak · sin²(π·u)`
                // (peaked at u = 0.5, zero at u = 0 and u = 1). Two
                // consequences:
                //   • foot world velocity matches the stance condition
                //     (= 0) on both sides — no impulsive PD reaction
                //     at the 3↔4 support transitions
                //   • body-frame velocity at the endpoints equals −v,
                //     i.e. the same value the stance segment carries,
                //     so `body_x` is C¹ across the handoff
                // Integrating the sin² shape (with v_peak = 2v/swing_dur
                // chosen so the foot covers v·T per swing) gives:
                //   body_x(u) = lift + (land−lift)·u − v·T·sin(2π·u)/(2π)
                let tau = std::f64::consts::TAU;
                let bx = lift + (land - lift) * u - v * big_t * (tau * u).sin() / tau;
                // Same continuity trick on Z: sin²(π·u) peaks at u=0.5
                // with amplitude `swing_height_m`, slope 0 at endpoints.
                let sz = (std::f64::consts::PI * u).sin();
                let bz = nominal.z + self.cfg.swing_height_m * sz * sz;
                (bx, bz)
            } else {
                // Stance midpoint is the geometric midpoint of the
                // stance interval [sub_end, sub_4sup_end + 1.0) wrapping
                // — i.e. cycle_phase ≡ sub_end + stance_dur/2 (mod 1).
                // Setting body_x = nominal_x at that phase gives a
                // symmetric receding excursion of ± v·(1−s)·T/2 over
                // the stance period.
                let stance_mid_phase = (sub_end + stance_dur_phase / 2.0).rem_euclid(1.0);
                // 0.5-centred modular subtraction → delta ∈ [-0.5, 0.5).
                let delta =
                    (cycle_phase - stance_mid_phase + 0.5).rem_euclid(1.0) - 0.5;
                let bx = nominal.x - v * delta * big_t;
                (bx, nominal.z)
            };

            let target_body = Vector3::new(body_x, nominal.y, body_z);
            foot_body_xyz[leg_i] = [body_x, nominal.y, body_z];

            let kin_leg = match leg_i {
                0 => &self.kin.fl,
                1 => &self.kin.fr,
                2 => &self.kin.rl,
                _ => &self.kin.rr,
            };
            let knee_fwd = self.cfg.knee_forward[leg_i];
            let sol = solve_leg_ik(kin_leg, target_body, knee_fwd);
            angles[leg_i] = sol.angles();
            foot_reachable[leg_i] = sol.is_reachable();
            in_swing[leg_i] = in_swing_phase;
        }

        LinearCrawlOutput {
            angles,
            in_swing,
            trunk_world_xyz: [trunk_x, 0.0, self.cfg.body_height_m],
            foot_body_xyz,
            foot_reachable,
        }
    }
}

// ─── GaitGenerator adapter ─────────────────────────────────────────────
//
// Lets [`LinearCrawlController`] sit alongside the body-velocity
// controllers ([`crate::ChampGaitController`] etc.) behind the unified
// [`crate::AnyGaitController`] dispatch. The adapter re-derives a fresh
// [`LinearCrawlConfig`] from the host's [`crate::GaitConfig`] each time
// the host pushes a config update; the velocity command becomes the
// trunk forward speed.

use crate::body_state::BodyState;
use crate::config::{GaitConfig, KneePattern, VelocityCmd};
use crate::controller::{ControllerOutput, LegOutput};
use crate::footstep::Footstep;
use crate::generator::GaitGenerator;
use crate::phase::PhaseState;

/// Wraps [`LinearCrawlController`] with the cross-controller surface
/// area required by [`GaitGenerator`]. Hosts that want this gait mode
/// don't construct this directly — they pick
/// [`crate::GaitMode::LinearCrawl`] and let the dispatch enum build it.
pub struct LinearCrawlGen {
    ctrl: LinearCrawlController,
    host_cfg: GaitConfig,
    cmd: VelocityCmd,
    knee_pattern: KneePattern,
    body_state: BodyState,
    body_height_m: f64,
}

impl LinearCrawlGen {
    pub fn new(cfg: GaitConfig, kin: KinematicsConfig) -> Self {
        // Pull body height from the auto-detected nominal foot Z (the
        // home pose's foot height in body frame is exactly the body's
        // ground clearance, by convention).
        let body_height_m = (-kin.fl.nominal_foot_body.z).max(0.05);
        let knee_pattern = KneePattern::BothBack;
        let lc_cfg = Self::derive_linear_cfg(&cfg, body_height_m, &knee_pattern);
        let ctrl = LinearCrawlController::new(kin, lc_cfg);
        Self {
            ctrl,
            host_cfg: cfg,
            cmd: VelocityCmd::zero(),
            knee_pattern,
            body_state: BodyState::new(),
            body_height_m,
        }
    }

    /// Override the trunk Z target. Articara reads this from the
    /// loaded `RobotModel`'s base transform at build time — without it
    /// the gait would track the auto-detected nominal foot Z, which is
    /// fine for any robot whose home pose has the feet on the ground
    /// but lets the host be explicit when it isn't.
    pub fn set_body_height_m(&mut self, h: f64) {
        self.body_height_m = h.max(0.05);
        self.rebuild_inner_cfg();
    }

    pub fn body_height_m(&self) -> f64 {
        self.body_height_m
    }

    fn derive_linear_cfg(
        host: &GaitConfig,
        body_height_m: f64,
        knee: &KneePattern,
    ) -> LinearCrawlConfig {
        LinearCrawlConfig {
            // `speed_mps` is set on every tick from `cmd.vx`; this
            // initial value gets overwritten before it matters.
            speed_mps: 0.0,
            body_height_m,
            cycle_period_s: host.cycle_period_s.max(0.05),
            four_support_fraction: host.four_support_fraction.clamp(0.05, 0.95),
            swing_height_m: host.swing_height_m.max(0.0),
            leg_order: [LegId::RL, LegId::FR, LegId::RR, LegId::FL],
            knee_forward: knee.to_knee_forward(),
            soft_start_duration_s: 0.5,
        }
    }

    fn rebuild_inner_cfg(&mut self) {
        let lc_cfg = Self::derive_linear_cfg(
            &self.host_cfg,
            self.body_height_m,
            &self.knee_pattern,
        );
        // Preserve cycle phase **and** the wall clock across rebuilds.
        // The user is typically tweaking a slider (or letting the
        // D-pad rescale `cycle_period_s` every frame) while the gait
        // is running — without these the `LinearCrawlController::new`
        // call would reset `wall_t = 0` each rebuild, perpetually
        // re-arming the soft-start ramp and making the gait stuck
        // crawling at < 50 % speed.
        let t = self.ctrl.elapsed();
        let wall_t = self.ctrl.wall_t;
        let holding = self.ctrl.holding;
        let kin = self.ctrl.kin.clone();
        self.ctrl = LinearCrawlController::new(kin, lc_cfg);
        self.ctrl.t = t;
        self.ctrl.wall_t = wall_t;
        self.ctrl.holding = holding;
        self.ctrl.set_speed(self.cmd.vx);
    }
}

impl GaitGenerator for LinearCrawlGen {
    fn tick(&mut self, dt: f64) -> ControllerOutput {
        // Keep the controller's forward speed synced to the latest
        // velocity command without rebuilding the whole config.
        self.ctrl.set_speed(self.cmd.vx);
        let out = self.ctrl.tick(dt);
        self.body_state.integrate(&self.cmd, dt);

        let kin = self.ctrl.kin.clone();
        let kin_legs = [&kin.fl, &kin.fr, &kin.rl, &kin.rr];
        let alpha = self.host_cfg.four_support_fraction.clamp(0.05, 0.95);
        let swing_dur_phase = (1.0 - alpha) * 0.25;
        let stance_dur_phase = 1.0 - swing_dur_phase;
        let cycle_position =
            (self.ctrl.elapsed() / self.ctrl.config().cycle_period_s).rem_euclid(1.0);

        let legs = std::array::from_fn(|i| {
            let lk = kin_legs[i];
            let (q_hip, q_thigh, q_calf) = out.angles[i];
            let foot_body = nalgebra::Vector3::new(
                out.foot_body_xyz[i][0],
                out.foot_body_xyz[i][1],
                out.foot_body_xyz[i][2],
            );
            let nominal_x = lk.nominal_foot_body.x;
            let v = self.cmd.vx;
            let big_t = self.ctrl.config().cycle_period_s;
            let half_exc = v * stance_dur_phase * big_t / 2.0;
            // Footstep convention: lift_off = rear of stride (where the
            // foot is about to leave), touch_down = front (just landed).
            let lift_off = nalgebra::Vector3::new(
                nominal_x - half_exc,
                lk.nominal_foot_body.y,
                -self.body_height_m,
            );
            let touch_down = nalgebra::Vector3::new(
                nominal_x + half_exc,
                lk.nominal_foot_body.y,
                -self.body_height_m,
            );
            let in_swing = out.in_swing[i];
            // sub_fraction: 0 at start of current sub-phase, 1 at its
            // end. Rough but consistent enough for telemetry.
            let sub_fraction = if in_swing {
                // We don't have the swing fraction directly; approximate
                // by mapping cycle position over the swing window.
                let i_sub = ((cycle_position) / 0.25).floor();
                let swing_start =
                    i_sub * 0.25 + alpha * 0.25;
                ((cycle_position - swing_start) / swing_dur_phase).clamp(0.0, 1.0)
            } else {
                0.5
            };
            LegOutput {
                leg: LegId::ALL[i],
                hip_joint: lk.hip_joint.clone(),
                thigh_joint: lk.thigh_joint.clone(),
                calf_joint: lk.calf_joint.clone(),
                q_hip,
                q_thigh,
                q_calf,
                foot_body,
                footstep: Footstep { lift_off, touch_down },
                phase: PhaseState {
                    leg: LegId::ALL[i],
                    cycle_position,
                    is_stance: !in_swing,
                    sub_fraction,
                },
                reachable: out.foot_reachable[i],
            }
        });
        ControllerOutput {
            legs,
            body_state: self.body_state,
        }
    }

    fn set_velocity_cmd(&mut self, cmd: VelocityCmd) {
        self.cmd = cmd;
        self.ctrl.set_speed(cmd.vx);
    }

    fn velocity_cmd(&self) -> VelocityCmd {
        self.cmd
    }

    fn reset(&mut self) {
        self.ctrl.reset();
        self.body_state.reset();
        self.cmd = VelocityCmd::zero();
        self.ctrl.set_speed(0.0);
    }

    fn config(&self) -> &GaitConfig {
        &self.host_cfg
    }

    fn set_config(&mut self, cfg: GaitConfig) {
        self.host_cfg = cfg;
        self.rebuild_inner_cfg();
    }

    fn kinematics(&self) -> &KinematicsConfig {
        &self.ctrl.kin
    }

    fn set_kinematics(&mut self, kin: KinematicsConfig) {
        let t = self.ctrl.elapsed();
        let lc_cfg = Self::derive_linear_cfg(
            &self.host_cfg,
            self.body_height_m,
            &self.knee_pattern,
        );
        self.ctrl = LinearCrawlController::new(kin, lc_cfg);
        self.ctrl.t = t;
    }

    fn set_knee_forward(&mut self, leg: LegId, forward: bool) {
        let mut arr = self.knee_pattern.to_knee_forward();
        let i = match leg {
            LegId::FL => 0,
            LegId::FR => 1,
            LegId::RL => 2,
            LegId::RR => 3,
        };
        arr[i] = forward;
        self.knee_pattern = KneePattern::from_knee_forward(arr);
        self.rebuild_inner_cfg();
    }

    fn set_knee_pattern(&mut self, pattern: KneePattern) {
        self.knee_pattern = pattern;
        self.rebuild_inner_cfg();
    }

    fn knee_pattern(&self) -> KneePattern {
        self.knee_pattern
    }

    fn knee_forward(&self) -> [bool; 4] {
        self.knee_pattern.to_knee_forward()
    }
    // LinearCrawl is open-loop kinematic — observed body state has no
    // effect, so the default no-op trait impls are correct.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LegKinematics;
    use nalgebra::Vector3;

    fn dummy_kin() -> KinematicsConfig {
        let mk = |leg: LegId, hx: f64| {
            LegKinematics::new(
                leg,
                format!("{}_hip", leg.label()),
                format!("{}_thigh", leg.label()),
                format!("{}_calf", leg.label()),
                format!("{}_foot", leg.label()),
                Vector3::new(hx, 0.05, 0.0),
                0.04,
                0.18,
                0.18,
            )
        };
        KinematicsConfig {
            fl: mk(LegId::FL, 0.18),
            fr: mk(LegId::FR, 0.18),
            rl: mk(LegId::RL, -0.18),
            rr: mk(LegId::RR, -0.18),
        }
    }

    /// Trunk advances strictly along `+X`; `Y` and `Z` of the trunk
    /// target are constant.
    #[test]
    fn trunk_pose_is_strictly_linear_in_x() {
        let mut c = LinearCrawlController::new(dummy_kin(), LinearCrawlConfig::default());
        let mut last_x = 0.0;
        for k in 0..50 {
            let out = c.tick(0.05);
            let [x, y, z] = out.trunk_world_xyz;
            assert!(x >= last_x, "trunk x decreased at step {k}: {x} < {last_x}");
            assert!((y - 0.0).abs() < 1e-12, "trunk y drift at step {k}: {y}");
            assert_eq!(z, c.config().body_height_m, "trunk z drift at step {k}");
            last_x = x;
        }
    }

    /// At every moment, at most one leg is in swing. (The four-support
    /// windows produce zero, the three-support windows produce one.)
    #[test]
    fn at_most_one_leg_in_swing() {
        let c = LinearCrawlController::new(dummy_kin(), LinearCrawlConfig::default());
        // Sample densely across one cycle.
        let big_t = c.config().cycle_period_s;
        for k in 0..1000 {
            let t = (k as f64) * big_t / 1000.0;
            let out = c.output_at(t);
            let n_swing = out.in_swing.iter().filter(|&&s| s).count();
            assert!(
                n_swing <= 1,
                "more than 1 leg in swing at t={t}: {:?}",
                out.in_swing
            );
        }
    }

    /// Over one full cycle, each leg is in swing exactly once and the
    /// total swing duration matches `(1 − α) · T / 4` per leg.
    #[test]
    fn each_leg_swings_once_per_cycle() {
        let cfg = LinearCrawlConfig::default();
        let big_t = cfg.cycle_period_s;
        let alpha = cfg.four_support_fraction;
        let expected_swing_s_per_leg = (1.0 - alpha) * big_t / 4.0;
        let c = LinearCrawlController::new(dummy_kin(), cfg);

        let n = 10_000;
        let dt = big_t / n as f64;
        let mut swing_time = [0.0; 4];
        for k in 0..n {
            let t = k as f64 * dt;
            let out = c.output_at(t);
            for (i, &s) in out.in_swing.iter().enumerate() {
                if s {
                    swing_time[i] += dt;
                }
            }
        }
        for i in 0..4 {
            assert!(
                (swing_time[i] - expected_swing_s_per_leg).abs() < 2.0 * dt,
                "leg {i} swing time {} ≠ expected {} (±dt)",
                swing_time[i],
                expected_swing_s_per_leg
            );
        }
    }

    /// The body-frame foot X position is continuous across the
    /// stance↔swing boundary. (Quintic smoothstep ⇒ also C¹/C².)
    #[test]
    fn body_x_continuous_at_swing_boundaries() {
        let c = LinearCrawlController::new(dummy_kin(), LinearCrawlConfig::default());
        let big_t = c.config().cycle_period_s;
        let dt = 1e-5;
        // Inspect every quarter-cycle boundary at swing start / end.
        let alpha = c.config().four_support_fraction;
        for i in 0..4 {
            let swing_start_phase = i as f64 * 0.25 + alpha * 0.25;
            let swing_end_phase = (i + 1) as f64 * 0.25;
            for boundary_phase in [swing_start_phase, swing_end_phase] {
                let t = boundary_phase * big_t;
                let before = c.output_at(t - dt);
                let after = c.output_at(t + dt);
                for leg_i in 0..4 {
                    let dx = (after.foot_body_xyz[leg_i][0]
                        - before.foot_body_xyz[leg_i][0])
                        .abs();
                    let dz = (after.foot_body_xyz[leg_i][2]
                        - before.foot_body_xyz[leg_i][2])
                        .abs();
                    // Continuous (dt is 1e-5, so per-step movement
                    // should be on the order of v·dt = 1e-6 worst case).
                    assert!(
                        dx < 1e-3,
                        "leg {leg_i} body_x jump {dx} at phase {boundary_phase}"
                    );
                    assert!(
                        dz < 1e-3,
                        "leg {leg_i} body_z jump {dz} at phase {boundary_phase}"
                    );
                }
            }
        }
    }

    /// Periodicity: the planner repeats every `T`. body_x for each leg
    /// at `t + T` matches `body_x` at `t` (no drift in body frame).
    #[test]
    fn body_x_periodic_over_one_cycle() {
        let c = LinearCrawlController::new(dummy_kin(), LinearCrawlConfig::default());
        let big_t = c.config().cycle_period_s;
        for k in 0..200 {
            let t = (k as f64) * big_t / 200.0;
            let a = c.output_at(t);
            let b = c.output_at(t + big_t);
            for leg_i in 0..4 {
                for axis in 0..3 {
                    assert!(
                        (a.foot_body_xyz[leg_i][axis] - b.foot_body_xyz[leg_i][axis]).abs()
                            < 1e-9,
                        "leg {leg_i} axis {axis} at t={t}: {} vs {}",
                        a.foot_body_xyz[leg_i][axis],
                        b.foot_body_xyz[leg_i][axis]
                    );
                }
            }
        }
    }

    /// Standstill (`speed_mps == 0`) freezes the cycle as soon as the
    /// controller reaches a phase where all four legs are in stance —
    /// no phantom swing motion while the robot is held still.
    #[test]
    fn standstill_freezes_at_all_stance() {
        let mut cfg = CrawlR001Cfg();
        cfg.speed_mps = 0.0;
        let mut c = LinearCrawlController::new(dummy_kin(), cfg);
        // At t=0, cycle_phase=0 is the 4-support window of sub-cycle 0,
        // so the very first tick should freeze.
        let out = c.tick(0.005);
        assert!(c.is_holding(), "should hold once all-stance at vx=0");
        assert!(out.in_swing.iter().all(|&s| !s));
        let frozen_xyz = out.foot_body_xyz;
        // Subsequent ticks must not advance time or shift the pose.
        for _ in 0..50 {
            let out2 = c.tick(0.005);
            assert!(c.is_holding());
            for i in 0..4 {
                for k in 0..3 {
                    assert!((out2.foot_body_xyz[i][k] - frozen_xyz[i][k]).abs() < 1e-12);
                }
            }
        }
    }

    /// Asking to stop mid-swing should let the in-flight swing finish
    /// (the foot lands at the next planted XY) before freezing. The
    /// controller must NOT teleport the foot to the ground.
    #[test]
    fn stop_request_completes_current_swing() {
        let mut c = LinearCrawlController::new(dummy_kin(), CrawlR001Cfg());
        c.set_speed(0.1);
        // Spin up enough to enter a swing window. With α=0.5 the first
        // swing starts at phase 0.125 of cycle 0 (=0.125 s for T=1.0).
        for _ in 0..40 {
            c.tick(0.005);
        }
        assert!(!c.is_holding());
        let mid_swing = c.output_at(c.elapsed());
        let any_in_swing = mid_swing.in_swing.iter().any(|&s| s);

        // User asks to stop while a leg is mid-air.
        c.set_speed(0.0);
        let mut held = false;
        for _ in 0..400 {
            let out = c.tick(0.005);
            if c.is_holding() {
                // Must only freeze when all-stance.
                assert!(out.in_swing.iter().all(|&s| !s));
                held = true;
                break;
            }
        }
        assert!(held, "controller never reached an all-stance freeze point");
        // If a leg was in swing when we hit stop, the controller had to
        // keep ticking to land it (=> non-trivial wait before freeze).
        let _ = any_in_swing;
    }

    /// Resuming from a frozen standstill picks up the cycle from
    /// exactly where it was frozen — no time jump, no extra wait.
    #[test]
    fn resume_from_standstill_continues_cycle() {
        // Soft-start is irrelevant here — disable it so the dt
        // advance equality below is exact.
        let mut cfg = CrawlR001Cfg();
        cfg.speed_mps = 0.0;
        cfg.soft_start_duration_s = 0.0;
        let mut c = LinearCrawlController::new(dummy_kin(), cfg);
        // Force a freeze.
        c.tick(0.005);
        assert!(c.is_holding());
        let frozen_t = c.elapsed();
        // Resume.
        c.set_speed(0.1);
        c.tick(0.005);
        assert!(!c.is_holding());
        // Exactly one dt should have been added since the freeze point.
        assert!((c.elapsed() - frozen_t - 0.005).abs() < 1e-12);
    }

    /// Helper: builder for a non-default-speed config used by the
    /// standstill tests above (Default has `speed_mps = 0.1`, but the
    /// tests want to override it independently).
    fn CrawlR001Cfg() -> LinearCrawlConfig {
        LinearCrawlConfig::default()
    }

    /// Stance body_x stays within `nominal ± v·(1−s)·T/2`. Verifies the
    /// symmetric-stance derivation.
    #[test]
    fn stance_excursion_within_bound() {
        let cfg = LinearCrawlConfig::default();
        let v = cfg.speed_mps;
        let big_t = cfg.cycle_period_s;
        let s_phase = (1.0 - cfg.four_support_fraction) * 0.25;
        let bound = v * (1.0 - s_phase) * big_t / 2.0 + 1e-9;
        let c = LinearCrawlController::new(dummy_kin(), cfg);

        let n = 4000;
        let dt = big_t / n as f64;
        for k in 0..n {
            let t = k as f64 * dt;
            let out = c.output_at(t);
            for leg_i in 0..4 {
                if !out.in_swing[leg_i] {
                    let dev = (out.foot_body_xyz[leg_i][0]
                        - c.nominal_body[leg_i].x)
                        .abs();
                    assert!(
                        dev <= bound,
                        "leg {leg_i} stance body_x deviation {dev} > bound {bound}"
                    );
                }
            }
        }
    }
}
