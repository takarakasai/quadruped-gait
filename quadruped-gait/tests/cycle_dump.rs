//! Integration test: simulate one full Trot cycle at vx = 0.3 m/s through
//! the full Phase 2 controller stack. Dumps a CSV of (time, body-frame
//! foot xyz, joint angles, body world pose) for every leg every tick so
//! the trajectory can be plotted and eyeballed.
//!
//! Originally written as a Phase 1 hand-wired smoke test; rewritten in
//! Phase 2 to drive [`GaitController`] directly. The pre-controller
//! Raibert / phase / IK invariants remain valid since the controller is
//! a deterministic composition of the same primitives.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use approx::assert_relative_eq;
use nalgebra::Vector3;
use quadruped_gait::{
    ControllerOutput, GaitConfig, GaitController, KinematicsConfig, LegId, LegKinematics,
    LegOutput, VelocityCmd,
};

/// Build a small symmetric quadruped: 0.36 m body length, 0.10 m body
/// width, 0.18 m upper / lower segments, 0.04 m lateral offset from the
/// roll axis to the pitch axis. Roughly a Mini Pupper / Solo footprint.
fn build_kinematics() -> KinematicsConfig {
    let body_half_len = 0.18;   // front hip at +x, rear at -x
    let body_half_width = 0.05; // hip lateral offset on body
    let lat = 0.04;
    let l1 = 0.18;
    let l2 = 0.18;

    let mk = |leg: LegId, sx: f64, sy: f64, lname: &str| {
        let mut kin = LegKinematics::new(
            leg,
            format!("{lname}_hip"),
            format!("{lname}_thigh"),
            format!("{lname}_calf"),
            format!("{lname}_foot"),
            Vector3::new(sx * body_half_len, sy * body_half_width, 0.0),
            lat,
            l1,
            l2,
        );
        // Override nominal: bent knee, ~30° from straight, so the leg has
        // room to swing forward/backward without hitting the L1+L2 limit.
        // Compute z so r_planar = 0.95 · (L1 + L2) at neutral.
        let neutral_extension = (l1 + l2) * 0.92; // 92% of full extension
        kin.nominal_foot_body.z = kin.hip_offset.z - neutral_extension;
        kin
    };

    KinematicsConfig {
        fl: mk(LegId::FL,  1.0,  1.0, "FL"),
        fr: mk(LegId::FR,  1.0, -1.0, "FR"),
        rl: mk(LegId::RL, -1.0,  1.0, "RL"),
        rr: mk(LegId::RR, -1.0, -1.0, "RR"),
    }
}

/// One sample of the dump CSV. Per-leg slots are in canonical
/// [`LegId::ALL`] order (FL, FR, RL, RR) — same layout the controller
/// already produces, no remapping needed.
#[derive(Clone, Debug)]
struct Sample {
    t: f64,
    body_x: f64,
    body_y: f64,
    body_yaw: f64,
    legs: [LegOutput; 4],
}

/// Position of `id` inside [`LegId::ALL`] for canonical-order indexing.
fn slot_of(id: LegId) -> usize {
    match id {
        LegId::FL => 0,
        LegId::FR => 1,
        LegId::RL => 2,
        LegId::RR => 3,
    }
}

/// Run one cycle through [`GaitController`] and return the samples plus
/// a flag indicating whether every leg's IK was reachable throughout.
fn simulate_cycle(
    kin: &KinematicsConfig,
    cfg: &GaitConfig,
    vx: f64,
    dt: f64,
    n_ticks: usize,
) -> (Vec<Sample>, bool) {
    let mut ctrl = GaitController::new(cfg.clone(), kin.clone());
    ctrl.set_velocity_cmd(VelocityCmd { vx, vy: 0.0, wz: 0.0 });
    let mut samples = Vec::with_capacity(n_ticks);
    let mut all_reachable = true;
    for tick in 0..n_ticks {
        let out: ControllerOutput = ctrl.tick(dt);
        if !out.all_reachable() {
            all_reachable = false;
        }
        samples.push(Sample {
            t: (tick as f64 + 1.0) * dt,
            body_x: out.body_state.world_position.x,
            body_y: out.body_state.world_position.y,
            body_yaw: out.body_state.world_yaw,
            legs: out.legs,
        });
    }
    (samples, all_reachable)
}

/// Write the dump as CSV. One row per tick:
/// `t, body_x, body_y, body_yaw, {LEG}_{xfoot,yfoot,zfoot,hip,thigh,calf,stance}` × 4.
fn write_csv(path: &PathBuf, samples: &[Sample]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    write!(f, "t,body_x,body_y,body_yaw")?;
    for tag in [LegId::FL, LegId::FR, LegId::RL, LegId::RR] {
        for col in ["xfoot", "yfoot", "zfoot", "hip", "thigh", "calf", "stance"] {
            write!(f, ",{}_{col}", tag.label())?;
        }
    }
    writeln!(f)?;
    for s in samples {
        write!(f, "{:.5},{:.5},{:.5},{:.5}", s.t, s.body_x, s.body_y, s.body_yaw)?;
        for leg in &s.legs {
            write!(
                f,
                ",{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{}",
                leg.foot_body.x,
                leg.foot_body.y,
                leg.foot_body.z,
                leg.q_hip,
                leg.q_thigh,
                leg.q_calf,
                if leg.phase.is_stance { 1 } else { 0 },
            )?;
        }
        writeln!(f)?;
    }
    Ok(())
}

#[test]
fn one_trot_cycle_at_vx_0_3() {
    let kin = build_kinematics();
    let cfg = GaitConfig::trot();
    let dt = 0.002; // 500 Hz, matches MuJoCo default
    let n_ticks = (cfg.cycle_period_s / dt).round() as usize;
    let vx = 0.3;
    let (samples, all_reachable) = simulate_cycle(&kin, &cfg, vx, dt, n_ticks);
    assert!(all_reachable, "every IK target should stay inside the workspace");
    assert_eq!(samples.len(), n_ticks);

    // Dump for visual inspection. The file lives in the OS temp dir so we
    // don't pollute the repo; the path is printed so the user can inspect
    // it (e.g. `head -20 $TMPDIR/gait_phase1_cycle.csv`).
    let path = std::env::temp_dir().join("gait_phase1_cycle.csv");
    write_csv(&path, &samples).expect("write csv");
    eprintln!("wrote {} samples → {}", samples.len(), path.display());

    // ── Invariants ──────────────────────────────────────────────────

    // Slot indices (canonical FL/FR/RL/RR layout, see `slot_of`).
    let fl = slot_of(LegId::FL);
    let fr = slot_of(LegId::FR);
    let rl = slot_of(LegId::RL);
    let rr = slot_of(LegId::RR);

    // 1. Trot phasing: at any tick, FL and RR share stance/swing state,
    //    and FR / RL share the opposite state.
    for (i, s) in samples.iter().enumerate() {
        assert_eq!(
            s.legs[fl].phase.is_stance, s.legs[rr].phase.is_stance,
            "tick {i}: FL and RR should be in the same sub-phase",
        );
        assert_eq!(
            s.legs[fr].phase.is_stance, s.legs[rl].phase.is_stance,
            "tick {i}: FR and RL should be in the same sub-phase",
        );
        assert_ne!(
            s.legs[fl].phase.is_stance, s.legs[fr].phase.is_stance,
            "tick {i}: diagonal pairs should be in opposite sub-phases",
        );
    }

    // 2. The cycle is balanced: each leg spends `duty * n_ticks` ticks
    //    in stance (within ±1 due to integer rounding).
    let expected_stance = (cfg.duty_factor * n_ticks as f64).round() as i64;
    for slot in 0..4 {
        let count = samples.iter().filter(|s| s.legs[slot].phase.is_stance).count() as i64;
        assert!(
            (count - expected_stance).abs() <= 1,
            "leg {slot}: stance count {count} not within 1 of expected {expected_stance}",
        );
    }

    // 3. Step length matches the catalogue: the foot's body-frame x range
    //    over a cycle should equal vx · stance_duration. We inspect FL.
    let fl_x: Vec<f64> = samples.iter().map(|s| s.legs[fl].foot_body.x).collect();
    let x_min = fl_x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = fl_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let observed_step = x_max - x_min;
    let expected_step = vx * cfg.cycle_period_s * cfg.duty_factor;
    assert_relative_eq!(observed_step, expected_step, epsilon = 1e-3);

    // 4. Swing legs lift above the stance plane. Track FL during its
    //    swing sub-phase and confirm the peak z exceeds the nominal.
    let fl_nominal_z = kin.fl.nominal_foot_body.z;
    let fl_swing_zmax = samples
        .iter()
        .filter(|s| !s.legs[fl].phase.is_stance)
        .map(|s| s.legs[fl].foot_body.z)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        fl_swing_zmax > fl_nominal_z + cfg.swing_height_m * 0.5,
        "expected FL swing peak above nominal+0.5·H, got {fl_swing_zmax} vs nominal {fl_nominal_z}",
    );

    // 5. During stance, FL's foot.z stays exactly at the nominal plane.
    for s in &samples {
        if s.legs[fl].phase.is_stance {
            assert_relative_eq!(s.legs[fl].foot_body.z, fl_nominal_z, epsilon = 1e-9);
        }
    }

    // 6. Joint angles stay sane.
    for s in &samples {
        for leg in &s.legs {
            assert!(leg.q_hip.abs() < 0.6, "hip out of range: {}", leg.q_hip);
            assert!(leg.q_thigh.abs() < 1.5, "thigh out of range: {}", leg.q_thigh);
            assert!(leg.q_calf.abs() < 1.5, "calf out of range: {}", leg.q_calf);
        }
    }

    // 7. Body integrator: after `n_ticks · dt` seconds the world position
    //    should match vx · t (no yaw, body stays on the +x axis).
    let last = samples.last().unwrap();
    let expected_x = vx * (n_ticks as f64) * 0.002;
    assert_relative_eq!(last.body_x, expected_x, epsilon = 1e-9);
    assert_relative_eq!(last.body_y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(last.body_yaw, 0.0, epsilon = 1e-12);
}

#[test]
fn zero_command_holds_static_pose() {
    // Velocity zero → every leg holds the nominal foot pose, every joint
    // angle stays put across the entire cycle.
    let kin = build_kinematics();
    let cfg = GaitConfig::trot();
    let dt = 0.002;
    let n_ticks = (cfg.cycle_period_s / dt).round() as usize;
    let (samples, all_reachable) = simulate_cycle(&kin, &cfg, 0.0, dt, n_ticks);
    assert!(all_reachable);

    // Pick the first tick's joint vector and confirm every later tick
    // matches it. With vx=0, the phase generator holds, the foot target
    // collapses to the nominal stance, and IK output is constant.
    let baseline: Vec<(f64, f64, f64)> = samples[0]
        .legs
        .iter()
        .map(|l| (l.q_hip, l.q_thigh, l.q_calf))
        .collect();
    for (i, s) in samples.iter().enumerate().skip(1) {
        for slot in 0..4 {
            let (h0, t0, c0) = baseline[slot];
            assert_relative_eq!(s.legs[slot].q_hip, h0, epsilon = 1e-12);
            assert_relative_eq!(s.legs[slot].q_thigh, t0, epsilon = 1e-12);
            assert_relative_eq!(s.legs[slot].q_calf, c0, epsilon = 1e-12);
            assert!(
                s.legs[slot].phase.is_stance,
                "tick {i} leg {slot} not in stance",
            );
        }
    }
}
