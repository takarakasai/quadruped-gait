//! Phase 1 integration test: simulate one full Trot cycle at vx = 0.3 m/s
//! by manually wiring `PhaseGenerator` + Raibert footstep heuristic +
//! `swing_position` / `stance_position` + per-leg `solve_leg_ik`. Dumps a
//! CSV of (time, body-frame foot xyz, joint angles) for every leg every
//! tick so the trajectory can be plotted and eyeballed.
//!
//! The footstep planner and `GaitController` are deliberately NOT used —
//! they're Phase 2's job. This test serves three purposes:
//!
//! 1. Confirm the Phase 1 pieces fit together cleanly when wired by hand.
//! 2. Document the intended pipeline so Phase 2's controller has a
//!    reference behaviour.
//! 3. Lock in regression assertions against numeric drift in any of
//!    `PhaseGenerator`, `swing_position`, or `solve_leg_ik`.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use approx::assert_relative_eq;
use nalgebra::Vector3;
use quadruped_gait::{
    solve_leg_ik, stance_position, swing_position, GaitConfig, KinematicsConfig, LegId,
    LegKinematics, PhaseGenerator, PhaseState, VelocityCmd,
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

/// Inline Raibert-style footstep target for one leg at the current phase.
///
/// During stance, the foot stays planted on the ground while the body
/// moves forward — equivalently, the foot moves *backward* relative to
/// the body by `vx · t`. We span from `+step/2` at touch-down to
/// `-step/2` at lift-off.
///
/// During swing, the foot lifts off at `-step/2` (where stance left it),
/// arcs over the body's path, and lands at `+step/2` ready for the next
/// stance.
fn foot_target(
    kin: &LegKinematics,
    phase: PhaseState,
    cycle_period: f64,
    duty: f64,
    swing_height: f64,
    vx: f64,
) -> Vector3<f64> {
    // Step length is the distance the body covers during one stance phase.
    let stance_duration = cycle_period * duty;
    let step = vx * stance_duration;

    let fwd = Vector3::new(1.0, 0.0, 0.0);
    let nominal = kin.nominal_foot_body;
    let lift_off = nominal - fwd * (step * 0.5); // back side of stance
    let touch_down = nominal + fwd * (step * 0.5); // front side of stance

    if phase.is_stance {
        // Stance: from touch_down → lift_off as sub-fraction grows.
        stance_position(touch_down, lift_off, phase.sub_fraction)
    } else {
        // Swing: from lift_off → touch_down as sub-fraction grows.
        swing_position(lift_off, touch_down, swing_height, phase.sub_fraction)
    }
}

/// One sample of the dump CSV.
///
/// Per-leg arrays are stored in canonical [`LegId::ALL`] order
/// (FL, FR, RL, RR) regardless of the order [`PhaseGenerator::legs`]
/// returns, so test assertions can index by a fixed convention.
#[derive(Clone, Copy, Debug)]
struct Sample {
    t: f64,
    foot_body: [Vector3<f64>; 4],
    joints: [(f64, f64, f64); 4], // (hip, thigh, calf)
    is_stance: [bool; 4],
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

/// Run one cycle and return the samples plus a flag indicating whether
/// every leg's IK solution was reachable throughout.
fn simulate_cycle(
    kin: &KinematicsConfig,
    cfg: &GaitConfig,
    vx: f64,
    dt: f64,
    n_ticks: usize,
) -> (Vec<Sample>, bool) {
    let mut phase_gen = PhaseGenerator::new(cfg.clone());
    let cmd = VelocityCmd { vx, vy: 0.0, wz: 0.0 };
    let mut samples = Vec::with_capacity(n_ticks);
    let mut all_reachable = true;

    for tick in 0..n_ticks {
        phase_gen.advance(dt, &cmd);
        let phases = phase_gen.legs();
        let mut foot_body = [Vector3::zeros(); 4];
        let mut joints = [(0.0, 0.0, 0.0); 4];
        let mut is_stance = [false; 4];

        for ps in phases.iter() {
            let kin_leg = kin.leg(ps.leg);
            let target = foot_target(
                kin_leg,
                *ps,
                cfg.cycle_period_s,
                cfg.duty_factor,
                cfg.swing_height_m,
                vx,
            );
            // Front legs knee-forward, rear legs knee-back is a common
            // quadruped convention. For symmetric testing of the IK
            // alone, use knee_forward = false on all legs; flip later if
            // a specific URDF demands.
            let sol = solve_leg_ik(kin_leg, target, false);
            if !sol.is_reachable() {
                all_reachable = false;
            }
            let slot = slot_of(ps.leg);
            foot_body[slot] = target;
            joints[slot] = sol.angles();
            is_stance[slot] = ps.is_stance;
        }
        samples.push(Sample {
            t: (tick as f64 + 1.0) * dt,
            foot_body,
            joints,
            is_stance,
        });
    }
    (samples, all_reachable)
}

/// Write the dump as CSV. One row per tick, 4 legs × (xfoot, yfoot,
/// zfoot, hip, thigh, calf, stance) in canonical FL/FR/RL/RR order
/// matching [`Sample`]'s slot layout.
fn write_csv(path: &PathBuf, samples: &[Sample]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    write!(f, "t")?;
    for tag in [LegId::FL, LegId::FR, LegId::RL, LegId::RR] {
        for col in ["xfoot", "yfoot", "zfoot", "hip", "thigh", "calf", "stance"] {
            write!(f, ",{}_{col}", tag.label())?;
        }
    }
    writeln!(f)?;
    for s in samples {
        write!(f, "{:.5}", s.t)?;
        for i in 0..4 {
            let p = s.foot_body[i];
            let (h, t, c) = s.joints[i];
            write!(
                f,
                ",{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{}",
                p.x, p.y, p.z, h, t, c, if s.is_stance[i] { 1 } else { 0 },
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
            s.is_stance[fl], s.is_stance[rr],
            "tick {i}: FL and RR should be in the same sub-phase",
        );
        assert_eq!(
            s.is_stance[fr], s.is_stance[rl],
            "tick {i}: FR and RL should be in the same sub-phase",
        );
        assert_ne!(
            s.is_stance[fl], s.is_stance[fr],
            "tick {i}: diagonal pairs should be in opposite sub-phases",
        );
    }

    // 2. The cycle is balanced: each leg spends `duty * n_ticks` ticks
    //    in stance (within ±1 due to integer rounding).
    let expected_stance = (cfg.duty_factor * n_ticks as f64).round() as i64;
    for slot in 0..4 {
        let count = samples.iter().filter(|s| s.is_stance[slot]).count() as i64;
        assert!(
            (count - expected_stance).abs() <= 1,
            "leg {slot}: stance count {count} not within 1 of expected {expected_stance}",
        );
    }

    // 3. Step length matches the catalogue: the foot's body-frame x range
    //    over a cycle should equal vx · stance_duration. We inspect FL.
    let fl_x: Vec<f64> = samples.iter().map(|s| s.foot_body[fl].x).collect();
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
        .filter(|s| !s.is_stance[fl])
        .map(|s| s.foot_body[fl].z)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        fl_swing_zmax > fl_nominal_z + cfg.swing_height_m * 0.5,
        "expected FL swing peak above nominal+0.5·H, got {fl_swing_zmax} vs nominal {fl_nominal_z}",
    );

    // 5. During stance, FL's foot.z stays exactly at the nominal plane
    //    (within rounding). This is the contract: stance line is in the
    //    z = nominal_z plane.
    for s in &samples {
        if s.is_stance[fl] {
            assert_relative_eq!(s.foot_body[fl].z, fl_nominal_z, epsilon = 1e-9);
        }
    }

    // 6. Joint angles stay sane. Hips < ±0.5 rad (lateral motion only
    //    from the hip-to-thigh offset, not from a yaw command), thighs
    //    and calves bounded by the geometry.
    for s in &samples {
        for (h, t, c) in s.joints {
            assert!(h.abs() < 0.6, "hip out of range: {h}");
            assert!(t.abs() < 1.5, "thigh out of range: {t}");
            assert!(c.abs() < 1.5, "calf out of range: {c}");
        }
    }
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
    let baseline = samples[0].joints;
    for (i, s) in samples.iter().enumerate().skip(1) {
        for slot in 0..4 {
            let (h0, t0, c0) = baseline[slot];
            let (h, t, c) = s.joints[slot];
            assert_relative_eq!(h, h0, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(t, t0, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(c, c0, epsilon = 1e-12, max_relative = 1e-12);
            // Also: every leg must be in stance.
            assert!(s.is_stance[slot], "tick {i} leg {slot} not in stance");
        }
    }
}
