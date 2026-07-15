# Media

- `trot_mpc_horizon.mp4` — visualization of the SRBD trot-MPC
  cross-backend benchmark (`srbd_mpc::tests::mpc_backend_bench`,
  `ref/wbc_comparison.md` Sec.5p in the articara repo): the predicted
  10-step horizon body trajectory + per-foot GRF vectors + stance/swing
  state for the representative trot snapshot the benchmark solves.
  Rendered as a simple rigid-body box (the SRBD model's own level of
  fidelity — no leg kinematics are part of this MPC, so a box is the
  honest representation, not a simplified one) + point-mass GRF arrows,
  since every backend agrees on the answer (only solve time differs —
  see Sec.5p) there's nothing to visually compare between backends,
  only one representative trajectory (Clarabel's) is shown.
- `render_trot_mpc_horizon.py` — regenerates the video from
  `trot_mpc_horizon.csv`, which the benchmark test writes when run
  with `MPC_BENCH_CSV_OUT=<path> cargo test --release -p quadruped-gait
  srbd_mpc::tests::mpc_backend_bench -- --ignored --nocapture`.
