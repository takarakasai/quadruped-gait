# Media

- `trot_mpc_horizon.mp4` — visualization of the SRBD trot-MPC
  cross-backend benchmark (`srbd_mpc::tests::mpc_backend_bench`,
  `ref/wbc_comparison.md` Sec.5p in the articara repo): the predicted
  10-step horizon body trajectory + per-foot GRF vectors + stance/swing
  state for the representative trot snapshot the benchmark solves.
  Since every backend agrees on the answer (only solve time differs —
  see Sec.5p) there's nothing to visually compare between backends;
  only one representative trajectory (Clarabel's) is shown.

  Rendered with the real Go2 trunk mesh (`base_0..4.obj`, from the
  sibling `articara/models/unitree_go2/go2.misa`) positioned/oriented
  per the SRBD-predicted body pose — legs are still not drawn (the
  SRBD model has no leg kinematics at all, only a rigid trunk + point-
  mass GRFs at the foot offsets), so a bare trunk is the model's own
  level of fidelity, not a simplification of a fancier one. An earlier
  version used a generic box instead of the real mesh; swapped out
  once it was pointed out that "trot" should actually look like Go2.
- `render_trot_mpc_horizon.py` — regenerates the video. Needs two
  inputs from *two* repos:
  1. `trot_mpc_horizon.csv` — this repo's `mpc_backend_bench` writes it
     when run with `MPC_BENCH_CSV_OUT=<path> cargo test --release -p
     quadruped-gait srbd_mpc::tests::mpc_backend_bench -- --ignored
     --nocapture`.
  2. `go2_mesh_manifest.csv` — the *misa-wbc* repo's
     `go2_leg_singularity_demo` example writes this (parent joint +
     resolved mesh path + placement for every Go2 visual mesh); only
     the 5 `base_*` (trunk) entries are used here.
