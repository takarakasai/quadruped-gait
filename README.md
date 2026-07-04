# quadruped-gait

[![CI](https://github.com/takarakasai/quadruped-gait/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/takarakasai/quadruped-gait/actions/workflows/ci.yml) [![coverage](https://codecov.io/gh/takarakasai/quadruped-gait/graph/badge.svg)](https://codecov.io/gh/takarakasai/quadruped-gait)

Quadruped locomotion stack for [articara](https://github.com/takarakasai/articara)
and [go2-gait-runner](https://github.com/takarakasai/go2-gait-runner):

- **`quadruped-gait`** — gait generation library (CHAMP-equivalent trot /
  linear crawl / SRBD / centroidal / full-centroidal MPC), built on
  [misarta](https://github.com/takarakasai/misarta) primitives. The public
  front door is `viz` / `viz_sub` / `wbc` plus curated root re-exports;
  research knobs are exposed through the metadata-driven `exp` surface.
- **`legged-estimation`** — state estimation for legged robots: 18-state
  linear Kalman filter (legged_control port), IMU attitude estimation,
  contact-based leg odometry. Simulator- and GUI-independent.

Split out of the articara repo with history (2026-07). The MuJoCo-based
walk-regression suite lives in articara, which drives these crates as a
consumer — verify behavioural changes there before pushing.

## License

Apache-2.0
