//! Deploy-side runtime for RL locomotion policies exported to ONNX.
//!
//! One crate owns the policy I/O plumbing — observation assembly (Isaac
//! joint order, projected gravity), inference (pure-Rust `tract`), action →
//! joint-target mapping with hardware limits, a plausibility screen for live
//! observations, and full-fidelity CSV logging — so the Go2 hardware runner
//! (`go2-gait-runner`) and the MuJoCo sim-deploy path (articara) drive the
//! **identical** code. A sim run therefore validates the exact bytes that
//! later run on the robot; only the sensor/actuator adapters differ.
//!
//! The policy contract (matching the Isaac Lab crawl training config):
//! 45-d proprioceptive observation → MLP → 12 joint-position offsets,
//! `q_des = default + ACTION_SCALE · action`, tracked by the on-board PD at
//! the trained gains, with inference at [`POLICY_HZ`].
//!
//! Observation layout (all Isaac joint order, no scaling / normalization):
//!
//! | slice     | content                              |
//! |-----------|--------------------------------------|
//! | `[0..3)`  | base angular velocity (body, rad/s)  |
//! | `[3..6)`  | projected gravity (unit, body frame) |
//! | `[6..9)`  | velocity command `[vx, vy, wz]`      |
//! | `[9..21)` | joint position − default             |
//! | `[21..33)`| joint velocity                       |
//! | `[33..45)`| previous action                      |
//!
//! `base_lin_vel` is intentionally absent — it is not measurable under
//! low-level control, so the policy is trained without it.

use tract_onnx::prelude::*;

/// Observation dimension of the deployed policy.
pub const N_OBS: usize = 45;
/// Action dimension (one position offset per joint).
pub const N_ACT: usize = 12;
/// Policy inference rate (Isaac decimation 4 × sim dt 0.005 s).
pub const POLICY_HZ: f64 = 50.0;

/// Go2 deployment constants: joint-order conversion tables, the trained
/// nominal pose / gains, command ranges, and hardware joint limits.
///
/// Isaac groups the 12 joints by TYPE (all hips, all thighs, all calves,
/// FL/FR/RL/RR within each); the Go2 SDK groups by LEG (FR, FL, RR, RL ×
/// hip/thigh/calf).
pub mod go2 {
    /// Go2 SDK motor index for each Isaac joint index (reorder a policy
    /// ACTION out).
    pub const ISAAC_TO_GO2: [usize; 12] = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8];
    /// Isaac joint index for each Go2 SDK motor index (build the OBSERVATION
    /// from measured state).
    pub const GO2_TO_ISAAC: [usize; 12] = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10];
    /// Default joint positions in **Isaac order** (the policy's nominal
    /// pose). The action is applied as `q_des = default + ACTION_SCALE ·
    /// action`.
    pub const DEFAULT_ISAAC: [f64; 12] = [
        0.1, -0.1, 0.1, -0.1, // hips: FL,FR,RL,RR
        0.8, 0.8, 1.0, 1.0, //   thighs: FL,FR,RL,RR
        -1.5, -1.5, -1.5, -1.5, // calves
    ];
    /// Isaac Lab `JointPositionActionCfg` scale (`use_default_offset=True`).
    pub const ACTION_SCALE: f64 = 0.5;
    /// On-board PD gains the policy was trained with (Go2 actuator cfg).
    pub const POLICY_KP: f32 = 25.0;
    pub const POLICY_KD: f32 = 0.5;
    /// Crawl command ranges the policy was trained on (m/s, m/s, rad/s).
    pub const CMD_VX_RANGE: (f64, f64) = (-0.3, 0.6);
    pub const CMD_VY_RANGE: (f64, f64) = (-0.3, 0.3);
    pub const CMD_WZ_RANGE: (f64, f64) = (-0.5, 0.5);
    /// Go2 joint limits (rad) from `go2.misa`, indexed hip/thigh/calf
    /// (`motor_index % 3`).
    pub const JOINT_LIMITS: [(f64, f64); 3] = [
        (-1.0472, 1.0472),   // hip
        (-1.5708, 3.4907),   // thigh
        (-2.7227, -0.83776), // calf
    ];
}

use go2::*;

/// Sensor snapshot the observation is built from, in **Go2 SDK conventions**
/// (motor order FR,FL,RR,RL × hip/thigh/calf; IMU quaternion `w,x,y,z`).
/// Hosts adapt their source to this: the hardware runner from `LowState`,
/// the sim-deploy path from MuJoCo sensors.
#[derive(Clone, Copy, Debug, Default)]
pub struct ObsInput {
    /// Base angular velocity, body frame (gyroscope), rad/s.
    pub gyro_rad_s: [f32; 3],
    /// Base orientation quaternion `(w, x, y, z)`, body → world.
    pub quat_wxyz: [f32; 4],
    /// Measured joint positions in Go2 motor order, rad.
    pub joint_q_go2: [f32; 12],
    /// Measured joint velocities in Go2 motor order, rad/s.
    pub joint_dq_go2: [f32; 12],
}

/// Assemble the 45-d observation. See the crate docs for the layout.
pub fn build_obs(inp: &ObsInput, cmd: &[f64; 3], last_action: &[f64; 12]) -> Vec<f32> {
    let mut obs = Vec::with_capacity(N_OBS);
    obs.extend_from_slice(&inp.gyro_rad_s);
    // projected gravity (unit, body frame) from the orientation quaternion
    // (w,x,y,z): gravity_b = R(q)^T · (0,0,-1) = -[third row of R].
    let q = &inp.quat_wxyz;
    let (w, x, y, z) = (q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    obs.push((2.0 * (w * y - x * z)) as f32);
    obs.push((-2.0 * (y * z + w * x)) as f32);
    obs.push((2.0 * (x * x + y * y) - 1.0) as f32);
    obs.push(cmd[0] as f32);
    obs.push(cmd[1] as f32);
    obs.push(cmd[2] as f32);
    // joint position (relative to default) and velocity, reordered to Isaac
    let mut jp = [0.0f32; 12];
    let mut jv = [0.0f32; 12];
    for gidx in 0..12 {
        let iidx = GO2_TO_ISAAC[gidx];
        jp[iidx] = inp.joint_q_go2[gidx] - DEFAULT_ISAAC[iidx] as f32;
        jv[iidx] = inp.joint_dq_go2[gidx];
    }
    obs.extend_from_slice(&jp);
    obs.extend_from_slice(&jv);
    for a in last_action.iter() {
        obs.push(*a as f32);
    }
    obs
}

/// Clamp a `[vx, vy, wz]` command to the ranges the policy was trained on.
pub fn clamp_cmd(mut c: [f64; 3]) -> [f64; 3] {
    c[0] = c[0].clamp(CMD_VX_RANGE.0, CMD_VX_RANGE.1);
    c[1] = c[1].clamp(CMD_VY_RANGE.0, CMD_VY_RANGE.1);
    c[2] = c[2].clamp(CMD_WZ_RANGE.0, CMD_WZ_RANGE.1);
    c
}

/// Map a policy action (Isaac order) to Go2-ordered joint targets:
/// `q_des = default + ACTION_SCALE · action`, clamped to the hardware
/// joint limits.
pub fn action_to_q_des_go2(action_isaac: &[f64; 12]) -> [f64; 12] {
    let mut q_des = [0.0f64; 12];
    for i in 0..12 {
        let q_isaac = DEFAULT_ISAAC[i] + ACTION_SCALE * action_isaac[i];
        let g = ISAAC_TO_GO2[i];
        let (lo, hi) = JOINT_LIMITS[g % 3];
        q_des[g] = q_isaac.clamp(lo, hi);
    }
    q_des
}

/// A loaded, optimized, runnable ONNX policy.
pub struct OnnxPolicy {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl OnnxPolicy {
    /// Load and optimize an exported policy; the input is pinned to
    /// `[1, N_OBS]` f32 and the output must be `[1, N_ACT]`.
    pub fn load(path: &str) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| format!("load onnx {path}: {e}"))?
            .with_input_fact(0, f32::fact([1, N_OBS]).into())
            .map_err(|e| format!("input fact: {e}"))?
            .into_optimized()
            .map_err(|e| format!("optimize: {e}"))?
            .into_runnable()
            .map_err(|e| format!("runnable: {e}"))?;
        Ok(Self { model })
    }

    /// One inference step: 45-d observation → 12-d action (Isaac order).
    pub fn infer(&self, obs: &[f32]) -> Result<[f64; 12], String> {
        if obs.len() != N_OBS {
            return Err(format!("obs length {} != {N_OBS}", obs.len()));
        }
        let input: Tensor = tract_ndarray::Array2::<f32>::from_shape_vec((1, N_OBS), obs.to_vec())
            .map_err(|e| format!("obs shape: {e}"))?
            .into();
        let out = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| format!("inference: {e}"))?;
        let view = out[0]
            .to_array_view::<f32>()
            .map_err(|e| format!("output view: {e}"))?;
        if view.len() != N_ACT {
            return Err(format!("expected {N_ACT} outputs, got {}", view.len()));
        }
        let mut action = [0.0f64; 12];
        for i in 0..12 {
            action[i] = view[[0, i]] as f64;
        }
        Ok(action)
    }
}

/// Plausibility screen for a 45-d observation. Returns the (static) names of
/// every violated check — sign/unit/order mistakes and NaNs show up here long
/// before they are debuggable from robot behaviour. Thresholds are generous:
/// anything flagged is *implausible*, not merely unusual.
pub fn obs_anomalies(obs: &[f32]) -> Vec<&'static str> {
    let mut v = Vec::new();
    if obs.len() != N_OBS {
        v.push("bad_len");
        return v;
    }
    if obs.iter().any(|x| !x.is_finite()) {
        v.push("non_finite");
    }
    let g_norm = (obs[3] * obs[3] + obs[4] * obs[4] + obs[5] * obs[5]).sqrt();
    if !(0.7..=1.3).contains(&g_norm) {
        v.push("gravity_not_unit");
    }
    if obs[..3].iter().any(|x| x.abs() > 15.0) {
        v.push("gyro_out_of_range");
    }
    if obs[9..21].iter().any(|x| x.abs() > 1.8) {
        v.push("joint_pos_offset_large");
    }
    if obs[21..33].iter().any(|x| x.abs() > 35.0) {
        v.push("joint_vel_large");
    }
    v
}

/// CSV header for per-inference-tick logging: full observation and action so
/// sign / ordering questions can be settled offline.
pub fn csv_header() -> String {
    let mut h = String::from("t_s,mode,infer_us");
    for n in [
        "ang_vel_x", "ang_vel_y", "ang_vel_z", "grav_x", "grav_y", "grav_z", "cmd_vx", "cmd_vy",
        "cmd_wz",
    ] {
        h.push(',');
        h.push_str(n);
    }
    for p in ["jp_isaac", "jv_isaac", "prev_act", "act", "qdes_go2"] {
        for i in 0..12 {
            h.push_str(&format!(",{p}_{i}"));
        }
    }
    h.push_str(",anomalies");
    h
}

/// One CSV row matching [`csv_header`].
#[allow(clippy::too_many_arguments)]
pub fn write_csv_row<W: std::io::Write>(
    w: &mut W,
    t_s: f64,
    mode: &str,
    infer_us: u128,
    obs: &[f32],
    action: &[f64; 12],
    q_des_go2: &[f64; 12],
    anomalies: &[&str],
) -> Result<(), String> {
    let mut row = format!("{t_s:.4},{mode},{infer_us}");
    for x in obs {
        row.push_str(&format!(",{x:.6}"));
    }
    for x in action {
        row.push_str(&format!(",{x:.6}"));
    }
    for x in q_des_go2 {
        row.push_str(&format!(",{x:.6}"));
    }
    row.push(',');
    row.push_str(&anomalies.join("|"));
    writeln!(w, "{row}").map_err(|e| format!("csv write: {e}"))
}

/// Offline model validation — no robot, no sim. Checks shapes (45 → 12),
/// measures inference latency against the [`POLICY_HZ`] slot budget, and runs
/// a bounded-response probe over plausible pseudo-random observations.
pub fn selftest(model_path: &str) -> Result<(), String> {
    let policy = OnnxPolicy::load(model_path)?;
    eprintln!("selftest: loaded {model_path} OK (obs={N_OBS})");
    for (label, obs) in [("zeros", vec![0.0f32; N_OBS]), ("ones", vec![1.0f32; N_OBS])] {
        let action = policy.infer(&obs)?;
        let a: Vec<f64> = action.iter().map(|v| (v * 1000.0).round() / 1000.0).collect();
        eprintln!("selftest: action[{label}] = {a:?}");
    }

    // ── latency: must fit comfortably inside one inference slot ────────────
    let zeros = vec![0.0f32; N_OBS];
    for _ in 0..20 {
        policy.infer(&zeros)?;
    }
    let mut lat_us: Vec<u128> = Vec::with_capacity(200);
    for _ in 0..200 {
        let t0 = std::time::Instant::now();
        policy.infer(&zeros)?;
        lat_us.push(t0.elapsed().as_micros());
    }
    lat_us.sort_unstable();
    let mean = lat_us.iter().sum::<u128>() as f64 / lat_us.len() as f64;
    let p99 = lat_us[lat_us.len() * 99 / 100 - 1];
    let budget_us = (1e6 / POLICY_HZ) as u128;
    eprintln!(
        "selftest: latency mean {mean:.0}us p99 {p99}us max {}us (slot budget {budget_us}us)",
        lat_us[lat_us.len() - 1]
    );
    if p99 > budget_us / 2 {
        eprintln!(
            "selftest: WARNING — p99 uses more than half the inference slot; \
             expect jitter on the Go2's weaker CPU."
        );
    }

    // ── bounded-response probe: plausible pseudo-random obs (no rand dep) ──
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // uniform in [-1, 1)
        (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let mut abs_max = 0.0f64;
    let mut clamped = 0usize;
    for _ in 0..100 {
        let mut obs = vec![0.0f32; N_OBS];
        for v in obs.iter_mut().take(3) {
            *v = (next() * 2.0) as f32; // ang_vel ±2 rad/s
        }
        // random unit gravity direction, biased downward
        let (gx, gy, gz) = (next(), next(), next() - 1.0);
        let n = (gx * gx + gy * gy + gz * gz).sqrt().max(1e-9);
        obs[3] = (gx / n) as f32;
        obs[4] = (gy / n) as f32;
        obs[5] = (gz / n) as f32;
        obs[6] = (next() * CMD_VX_RANGE.1) as f32;
        obs[7] = (next() * CMD_VY_RANGE.1) as f32;
        obs[8] = (next() * CMD_WZ_RANGE.1) as f32;
        for i in 9..21 {
            obs[i] = (next() * 0.5) as f32; // joint offsets ±0.5 rad
        }
        for i in 21..33 {
            obs[i] = (next() * 3.0) as f32; // joint vel ±3 rad/s
        }
        for i in 33..45 {
            obs[i] = next() as f32; // previous action ±1
        }
        let action = policy.infer(&obs)?;
        for (i, a) in action.iter().enumerate() {
            if !a.is_finite() {
                return Err(format!("probe: non-finite action[{i}]"));
            }
            abs_max = abs_max.max(a.abs());
            let q = DEFAULT_ISAAC[i] + ACTION_SCALE * a;
            let (lo, hi) = JOINT_LIMITS[ISAAC_TO_GO2[i] % 3];
            if q < lo || q > hi {
                clamped += 1;
            }
        }
    }
    eprintln!(
        "selftest: bounded-response probe (100 plausible random obs): \
         |action|max = {abs_max:.2} ({clamped}/1200 joint targets would hit the \
         hardware limit clamp)"
    );
    if abs_max > 6.0 {
        eprintln!(
            "selftest: WARNING — actions exceed 6.0; with ACTION_SCALE {ACTION_SCALE} \
             that is a large q_des swing. Double-check obs scaling conventions."
        );
    }

    eprintln!("selftest: OK — model loads and infers ({N_OBS} -> {N_ACT}).");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_layout_zero_input() {
        let inp = ObsInput::default();
        let cmd = [0.1, -0.2, 0.3];
        let mut last = [0.0f64; 12];
        last[0] = 0.5;
        last[11] = -0.5;
        let obs = build_obs(&inp, &cmd, &last);
        assert_eq!(obs.len(), N_OBS);
        assert_eq!(&obs[..3], &[0.0, 0.0, 0.0]);
        // zero quaternion hits the gz = 2(x²+y²)−1 = −1 branch: [0,0,−1]
        assert_eq!(&obs[3..6], &[0.0, 0.0, -1.0]);
        assert!((obs[6] - 0.1).abs() < 1e-6);
        assert!((obs[7] + 0.2).abs() < 1e-6);
        assert!((obs[8] - 0.3).abs() < 1e-6);
        // q = 0 ⇒ joint-pos slice is −default (Isaac order)
        for i in 0..12 {
            assert!((obs[9 + i] as f64 + DEFAULT_ISAAC[i]).abs() < 1e-6, "jp[{i}]");
            assert_eq!(obs[21 + i], 0.0, "jv[{i}]");
        }
        assert!((obs[33] - 0.5).abs() < 1e-6);
        assert!((obs[44] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn isaac_go2_tables_are_mutual_inverses() {
        for g in 0..12 {
            assert_eq!(ISAAC_TO_GO2[GO2_TO_ISAAC[g]], g, "go2 {g}");
        }
        for i in 0..12 {
            assert_eq!(GO2_TO_ISAAC[ISAAC_TO_GO2[i]], i, "isaac {i}");
        }
    }

    #[test]
    fn joint_reorder_lands_each_motor_in_its_isaac_slot() {
        let mut inp = ObsInput::default();
        for g in 0..12 {
            inp.joint_q_go2[g] = (DEFAULT_ISAAC[GO2_TO_ISAAC[g]] + g as f64 * 0.01) as f32;
            inp.joint_dq_go2[g] = g as f32;
        }
        let obs = build_obs(&inp, &[0.0; 3], &[0.0; 12]);
        for g in 0..12 {
            let i = GO2_TO_ISAAC[g];
            assert!(
                (obs[9 + i] as f64 - g as f64 * 0.01).abs() < 1e-5,
                "jp motor {g} -> isaac {i}"
            );
            assert!((obs[21 + i] as f64 - g as f64).abs() < 1e-6, "jv {g}");
        }
    }

    #[test]
    fn gravity_projection_matches_nalgebra_reference() {
        use nalgebra::{Quaternion, UnitQuaternion, Vector3};
        // (w, x, y, z) — identity, 90° about each axis, and two arbitrary.
        let quats = [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071068, 0.7071068, 0.0, 0.0],
            [0.7071068, 0.0, 0.7071068, 0.0],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.4, 0.3, 0.5, 0.2],
            [-0.6, 0.1, -0.2, 0.5],
        ];
        for q in quats {
            let mut inp = ObsInput::default();
            let n = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
            for k in 0..4 {
                inp.quat_wxyz[k] = (q[k] / n) as f32;
            }
            let obs = build_obs(&inp, &[0.0; 3], &[0.0; 12]);
            let uq = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
            let g_ref = uq.inverse_transform_vector(&Vector3::new(0.0, 0.0, -1.0));
            for k in 0..3 {
                assert!(
                    (obs[3 + k] as f64 - g_ref[k]).abs() < 1e-5,
                    "quat {q:?} axis {k}: got {} want {}",
                    obs[3 + k],
                    g_ref[k]
                );
            }
        }
    }

    #[test]
    fn action_mapping_defaults_and_clamps() {
        // zero action ⇒ exactly the default pose, reordered to Go2
        let q = action_to_q_des_go2(&[0.0; 12]);
        for i in 0..12 {
            assert!(
                (q[ISAAC_TO_GO2[i]] - DEFAULT_ISAAC[i]).abs() < 1e-12,
                "isaac {i}"
            );
        }
        // huge action ⇒ every joint pinned to its hardware limit
        let q = action_to_q_des_go2(&[100.0; 12]);
        for (g, v) in q.iter().enumerate() {
            assert_eq!(*v, JOINT_LIMITS[g % 3].1, "motor {g} hi clamp");
        }
        let q = action_to_q_des_go2(&[-100.0; 12]);
        for (g, v) in q.iter().enumerate() {
            assert_eq!(*v, JOINT_LIMITS[g % 3].0, "motor {g} lo clamp");
        }
    }

    #[test]
    fn clamp_cmd_applies_trained_ranges() {
        let c = clamp_cmd([10.0, -10.0, 10.0]);
        assert_eq!(c, [CMD_VX_RANGE.1, CMD_VY_RANGE.0, CMD_WZ_RANGE.1]);
        let c = clamp_cmd([0.1, 0.1, -0.1]);
        assert_eq!(c, [0.1, 0.1, -0.1]);
    }

    #[test]
    fn obs_anomaly_screen_flags_the_right_things() {
        let clean = build_obs(&ObsInput::default(), &[0.0; 3], &[0.0; 12]);
        assert!(obs_anomalies(&clean).is_empty());

        let mut o = clean.clone();
        o[10] = f32::NAN;
        assert!(obs_anomalies(&o).contains(&"non_finite"));

        let mut o = clean.clone();
        o[3] = 0.0;
        o[4] = 0.0;
        o[5] = -0.2;
        assert!(obs_anomalies(&o).contains(&"gravity_not_unit"));

        let mut o = clean.clone();
        o[1] = 100.0;
        assert!(obs_anomalies(&o).contains(&"gyro_out_of_range"));

        let mut o = clean.clone();
        o[25] = 100.0;
        assert!(obs_anomalies(&o).contains(&"joint_vel_large"));

        assert_eq!(obs_anomalies(&[0.0; 3]), vec!["bad_len"]);
    }

    #[test]
    fn csv_row_matches_header_width() {
        let header = csv_header();
        let cols = header.split(',').count();
        assert_eq!(cols, 3 + 45 + 12 + 12 + 1);

        let obs = build_obs(&ObsInput::default(), &[0.0; 3], &[0.0; 12]);
        let mut buf: Vec<u8> = Vec::new();
        write_csv_row(&mut buf, 1.25, "shadow", 123, &obs, &[0.0; 12], &[0.0; 12], &["x"])
            .unwrap();
        let row = String::from_utf8(buf).unwrap();
        assert_eq!(row.trim_end().split(',').count(), cols);
        assert!(row.starts_with("1.2500,shadow,123,"));
    }
}
