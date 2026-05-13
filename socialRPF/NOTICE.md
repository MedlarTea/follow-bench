# NOTICE

`socialRPF` (the planning / benchmarking package shipped under
`follow-bench/socialRPF/`) is released under the MIT License,
Copyright (c) 2026 Hanjing YE — see `LICENSE`.

This package is built on top of, and bundles unmodified or adapted code from,
the following upstream MIT-licensed projects authored by Han Ruihua. Their
original copyright notices and permission notices are preserved verbatim in
`LICENSE-RDA-planner` (RDA-planner) and in `../ir-sim/LICENSE` (ir-sim), as
required by the MIT License. We gratefully acknowledge their contribution.

## Upstream projects

### RDA-planner (https://github.com/hanruihua/RDA-planner)
- Original copyright: Copyright (c) 2023 Han Ruihua, MIT License
- Citation: Han et al., "RDA: An Accelerated Collision Free Motion Planner
  for Autonomous Navigation in Cluttered Environments", IEEE RA-L 2023,
  doi:10.1109/LRA.2023.3242138
- Code from this project is bundled in `socialRPF/RDA_planner/`
  (the inner Python package that implements the ADMM-based RDA solver and
  its MPC variants `mpc_chasing_point*.py` / `mpc_chasing_traj.py`). The
  package keeps its original name `RDA_planner` to make the provenance
  visible in every `from RDA_planner.X import Y` import site.
- Some example scripts in `socialRPF/example/` (especially
  `path_track/`, `corridor/`, `reverse/`, `lidar_nav/`, `dynamic_obs/`)
  are adapted from the upstream `example/` directory.

### ir-sim (https://github.com/hanruihua/ir-sim)
- Original copyright: Copyright (c) 2022 Han Ruihua, MIT License
- Bundled in the sibling directory `follow-bench/ir-sim/`. We use it as the
  underlying simulator for the benchmark and have applied light
  modifications (recording `alg_cost_t`, lidar `build_map`, evaluation
  hooks, etc.). All upstream notices remain intact in `ir-sim/LICENSE`.

## New contributions in this fork (Hanjing YE)

The following directories / files are new work authored under the same
MIT license as `LICENSE`:

- `socialRPF/BSO_HFC_planner/` — Hybrid A* + uniform B-spline + OSQP MPC
  pipeline for person following.
- `socialRPF/follow_ahead_reaction/` — MCTS + RL value model + LSTM
  human-action prior follow-ahead planner and its runtime variants.
- `socialRPF/example/robot_person_following/` — the eight `*_diff.py` /
  `RL_diff.py` benchmark entry scripts, the `demo_scripts/`,
  `evaluate_scripts/`, and `visualize_scripts/` tooling, plus
  `global_params.py`.
- `socialRPF/dynamic_scenarios/` and
  `socialRPF/layout_scenarios/` — pre-generated YAML scenario
  configurations consumed by the benchmark (dynamic crowds and static
  layouts respectively).
- `socialRPF/traj_predictor/` — CV / CVKF / SGAN trajectory prediction
  glue (SGAN portions adapted from Social-GAN under its original license,
  see file headers).
- `socialRPF/DWA_planner/`, `socialRPF/SFM_planner/` — clean-room
  implementations of the DWA and Social-Force baselines used in the
  benchmark.

## How to cite

If you use `socialRPF` / `follow-bench` in academic work, please cite both
the underlying RDA-planner paper and our benchmark paper. See `README.md`
for BibTeX entries.
