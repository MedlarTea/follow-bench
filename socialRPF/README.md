# socialRPF

The Python package backing **Follow-Bench: A Unified Motion Planning Benchmark
for Socially-Aware Robot Person Following** (Hanjing YE, 2026).

For installation, evaluation pipelines, and citation info, see the top-level
[`README.md`](../README.md).

## Layout

```
socialRPF/
├── RDA_planner/                 # ← code from upstream hanruihua/RDA-planner
├── BSO_HFC_planner/             # local: Hybrid A* + B-spline + OSQP MPC
├── DWA_planner/, SFM_planner/   # local: classical baselines
├── traj_predictor/              # CV / CVKF / SGAN
├── follow_ahead_reaction/       # local: MCTS + RL value + LSTM prior
├── dynamic_scenarios/  # ORCA / SFM dynamic-crowd scenarios
├── layout_scenarios/   # corridor / intersection / doorway / clutter
├── example/robot_person_following/   # eight `*_diff.py` benchmark entrypoints
├── LICENSE                      # MIT, © 2026 Hanjing YE  (this work)
├── LICENSE-RDA-planner          # upstream MIT, © 2023 Han Ruihua
└── NOTICE.md                    # full provenance
```

## Upstream attribution

`socialRPF/RDA_planner/` is a vendored copy of
[hanruihua/RDA-planner](https://github.com/hanruihua/RDA-planner) (MIT,
© 2023 Han Ruihua). Its name is intentionally preserved so that every
`from RDA_planner.X import Y` import site documents the upstream source.

The upstream README is preserved as
[`README-RDA-planner.md`](README-RDA-planner.md) for historical reference.

The simulator under `follow-bench/ir-sim/` is from
[hanruihua/ir-sim](https://github.com/hanruihua/ir-sim) (MIT,
© 2022 Han Ruihua), with light modifications. See
[`NOTICE.md`](NOTICE.md) for the full provenance breakdown.
