# policy-runtime

Deploy-side runtime for the Go2 RL locomotion policy (ONNX). This crate is
the **single implementation** of the policy I/O — the hardware runner
([go2-gait-runner]) and the MuJoCo sim-deploy (articara's `go2_policy_sim`
example) both drive it, so a sim run validates the exact code that later
runs on the robot.

**This README doubles as the handoff between the training PC (Isaac Lab)
and the deploy PC.** If you are training/exporting on another machine,
everything the deployment expects is specified here; if you change any of
it in training, change this crate to match and re-verify in sim.

[go2-gait-runner]: https://github.com/takarakasai/go2-gait-runner

## Deployment contract (what the exported policy must match)

### Observation — 45-d f32, **no scaling / normalization**

| slice      | content                                   | frame / unit        |
|------------|-------------------------------------------|---------------------|
| `[0..3)`   | base angular velocity                     | body frame, rad/s   |
| `[3..6)`   | projected gravity (unit vector)           | body frame          |
| `[6..9)`   | velocity command `[vx, vy, wz]`           | m/s, m/s, rad/s     |
| `[9..21)`  | joint position − default                  | Isaac order, rad    |
| `[21..33)` | joint velocity                            | Isaac order, rad/s  |
| `[33..45)` | previous action (raw network output)      | Isaac order         |

- `base_lin_vel` is **intentionally absent** (not measurable under
  low-level control) — train without it.
- Isaac joint order = grouped by TYPE: 4 hips (FL,FR,RL,RR), 4 thighs,
  4 calves. Conversion tables to/from the Go2 SDK motor order live in
  `go2::{ISAAC_TO_GO2, GO2_TO_ISAAC}` and are unit-tested as mutual
  inverses.

### Action — 12-d f32, Isaac order

`q_des = DEFAULT_ISAAC + 0.5 · action` (Isaac Lab
`JointPositionActionCfg`, `use_default_offset=True`, `scale=0.5`),
clamped to the Go2 hardware joint limits at deploy time.

### Rates / gains (must match the training env)

| item                | value                                    |
|---------------------|------------------------------------------|
| inference           | 50 Hz (Isaac decimation 4 × sim dt 0.005) |
| on-board PD         | kp = 25, kd = 0.5                        |
| default pose (Isaac)| hips ±0.1, thighs 0.8/0.8/1.0/1.0, calves −1.5 |
| command ranges      | vx ∈ [−0.3, 0.6], vy ∈ [±0.3], wz ∈ [±0.5] |

All of these are constants in [`src/lib.rs`](src/lib.rs) `go2` module —
that file is the ground truth, this table is a mirror.

### Export requirements

- ONNX, input `[1, 45]` f32 → output `[1, 12]` f32, single input/output.
- Runtime is [tract-onnx] 0.21 (pure Rust): stick to plain MLP ops
  (Gemm / Elu / Relu / Tanh …), opset ≤ 18 is known-good (17 tested).
- No external weights files — one self-contained `.onnx`.

[tract-onnx]: https://crates.io/crates/tract-onnx

## Verification pipeline on the deploy PC

Bring the `.onnx` over and escalate; each stage proves the next without
letting the policy move a robot:

```sh
# 1. offline: shapes, latency vs the 20 ms slot, bounded-response probe
go2-gait-runner policy x --model policy.onnx --selftest

# 2. sim walk (articara repo) — obs sign/order/scale mistakes fall HERE,
#    and the Isaac->MuJoCo sim-to-sim gap shows before the sim-to-real one
cargo xtask run --features mujoco,policy-sim --example go2_policy_sim -- \
    --model policy.onnx --vx 0.3 --csv sim.csv

# 3. robot, no inference: hold default pose, eyeball live obs (tilt test)
go2-gait-runner policy eth0 --model policy.onnx --hold --csv hold.csv

# 4. robot, SHADOW: inference runs + is logged, motors hold default pose
go2-gait-runner policy eth0 --model policy.onnx --shadow --csv shadow.csv

# 5. real run, short leash
go2-gait-runner policy eth0 --model policy.onnx --duration 5 --csv run.csv
```

The `--csv` schema is identical in sim and on hardware (full 45-d obs,
12-d action, commanded q_des, inference latency, anomaly flags), so one
offline analysis serves both. An observation plausibility screen
(non-finite / non-unit gravity / out-of-range gyro, joint offsets,
velocities) runs live in every mode and is summarized on exit.

## Status & plan (2026-07-05)

- [x] I/O plumbing unit-tested (obs layout, Isaac↔Go2 tables, gravity
      projection cross-checked against nalgebra, CSV shape) — this crate
- [x] Hardware runner consumes this crate (LowState adapter only)
- [x] Sim-deploy end-to-end verified with a small-weight dummy MLP:
      4 s upright hold, clean plausibility screen, inference ≈ 250 µs
- [ ] **Next: real Isaac Lab export walking in sim** (stage 2 above)
- [ ] Joint-sign jig (`--jog`): the Isaac↔Go2 joint *sign* convention is
      still assumed identical (no per-joint flip) — verify per joint at
      low kp before the first real run
- [ ] Hardware stages 3–5

Known open assumption: joint signs (above). Everything else in the
contract is enforced by code + tests on the deploy side.
