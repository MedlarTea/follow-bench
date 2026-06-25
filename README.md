# Follow-Bench

`follow-bench` ships two Python packages:

- **`socialRPF/`** — the planner / benchmark code
- **`ir-sim/`** — the underlying simulator


## Install
- python >= 3.9

A clean conda env only needs the two `pip install -e .` calls below; every
runtime dependency (numpy, scipy, cvxpy, osqp, opencv, torch, stable-baselines3,
treelib, gymnasium, ...) is declared in `socialRPF/setup.py` and
`ir-sim/pyproject.toml`.

```bash
conda create -n follow-bench python=3.10 -y
conda activate follow-bench

pip install -e socialRPF
pip install -e ir-sim

# test
cd socialRPF/example/robot_person_following
bash demo_scripts/test_all_planners.sh
```

`evaluate.sh` will then write logs under
`/data/ssd/follow_bench_logs/<topography|dynamic|...>-1.5m/<planner>/...` and
the visualization scripts will pick up exactly the same tree on read.

## Demo Run

```bash
conda activate follow-bench
bash socialRPF/example/robot_person_following/demo_scripts/run_demo.sh
# or, to redirect logs:
LOG_DIR=/data/follow_bench_logs bash socialRPF/example/robot_person_following/demo_scripts/run_demo.sh
```

## Evaluated Settings
### **Evaluation Settings for Planner Performance**

Evaluation settings for planner performance under varying levels of complexity. The number of humans (H), occupancy conditions (O), and following configurations (F) are systematically varied across environmental layout and crowd-dynamics scenarios to characterize complexity factors such as collision risk, occlusion risk, and socially appropriate distance maintenance. Each configuration is evaluated over 100 runs with different random seeds.

| Experiment        | Variable Type              | Values                                  | Fixed Conditions                                      |
|------------------|---------------------------|------------------------------------------|------------------------------------------------------|
| Target Trajectory | -                         | -                                        | -                                                    |
| Dynamic (H)       | Human number              | {5, 10, 15, 20, 25, 30}                  | -                                                    |
| Layout (H)        | Corridor humans           | {4, 12, 20, 28}                          | 5.6 m width                                          |
|                  | Intersection humans       | {2, 10, 18, 26}                          | 4.8 m width                                          |
|                  | Doorway humans            | {2, 6, 10, 14}                           | 2.8 m width                                          |
|                  | Cluttered space humans    | {10, 20, 30, 40}                         | 30 obstacles                                         |
| Layout (O)        | Corridor width            | {6.8, 6.2, 5.6, 5.0} m                   | 20 humans                                            |
|                  | Intersection width        | {6.0, 5.4, 4.8, 4.2} m                   | 18 humans                                            |
|                  | Doorway width             | {3.4, 3.1, 2.8, 2.5} m                   | 10 humans                                            |
|                  | Cluttered space obstacles | {10, 20, 30, 40}                         | 30 humans                                            |
| Dynamic (F)       | Following configuration   | Back / Side at {1.0, 1.5, 2.0, 2.5} m    | 20 humans                                            |
| Layout (F)        | Following configuration   | Back / Side at {1.0, 1.5, 2.0, 2.5} m    | Corridor: 20 humans; 5.6 m width                     |
|                  |                           |                                          | Intersection: 18 humans; 4.8 m width                 |
|                  |                           |                                          | Doorway: 10 humans; 2.8 m width                      |
|                  |                           |                                          | Cluttered space: 30 humans; 30 obstacles             |

## Evaluation
Recommended for headless servers:
```bash
export MPLCONFIGDIR=/tmp/matplotlib
# (optional) redirect logs / figures off the source tree
export FOLLOWBENCH_LOG_DIR=/data/follow_bench_logs
export FOLLOWBENCH_EVAL_RESULTS=/data/follow_bench_figures
```

All scripts below accept `--headless` to disable window visualization and speed
up evaluation. The sweep yamls (`config_*_eval*.yaml`) declare their
`config_dir` and `log_dir` as repo-root-relative paths, so no editing is needed
on a fresh clone:

```bash
cd socialRPF/example/robot_person_following/evaluate_scripts
# Eval `Target Trajectory` (default 1.5m, including back and left_side following)
bash evaluate.sh config_target_traj_eval.yaml
# bash evaluate.sh config_target_traj_eval.yaml --headless

# Eval `corridor` in `Layout (H)` and `Layout (O)` (default 1.5m, including back and left_side following)
bash evaluate.sh config_topography_corridor_eval.yaml

# Eval `intersection` in `Layout (H)` and `Layout (O)` (default 1.5m, including back and left_side following)
bash evaluate.sh config_topography_intersection_eval.yaml

# Eval `doorway` in `Layout (H)` and `Layout (O)` (default 1.5m, including back and left_side following)
bash evaluate.sh config_topography_doorway_eval.yaml

# Eval `clustered space obstacles` in `Layout (H)` and `Layout (O)` (default 1.5m, including back and left_side following)
bash evaluate.sh config_topography_clutter_eval.yaml

# Eval Dynamic (H), back and side, following distance as 1.5m
bash evaluate.sh config_dynamic_crowd_eval.yaml

# Eval Dynamic (F), back and side, following distance as {1.0, 1.5, 2.0, 2.5}, 20 humans
bash evaluate.sh config_dynamic_crowd_eval_dynamicF.yaml 1.0
# bash evaluate.sh config_dynamic_crowd_eval_dynamicF.yaml 1.5
bash evaluate.sh config_dynamic_crowd_eval_dynamicF.yaml 2.0
bash evaluate.sh config_dynamic_crowd_eval_dynamicF.yaml 2.5

# Eval Layout (F), back and side, following distance as {1.0, 1.5, 2.0, 2.5}
bash evaluate.sh config_topography_eval_layoutF.yaml 1.0
# bash evaluate.sh config_topography_eval_layoutF.yaml 1.5
bash evaluate.sh config_topography_eval_layoutF.yaml 2.0
bash evaluate.sh config_topography_eval_layoutF.yaml 2.5

# (Optional) Dynamic crowd with SFM (instead of ORCA) human behaviour, used by
# `visualize_scripts/visualize_results_humans_sfm.py`. The four files split the
# baselines so several machines can run in parallel; on a single machine you
# can run them sequentially:
bash evaluate.sh config_dynamic_crowd_sfm_eval1.yaml
bash evaluate.sh config_dynamic_crowd_sfm_eval2.yaml
bash evaluate.sh config_dynamic_crowd_sfm_eval3.yaml
bash evaluate.sh config_dynamic_crowd_sfm_eval4.yaml
```

## Visualizing the results

After `evaluate.sh` finishes, the per-trial JSON results live under
`<repo>/logs/<experiment>-<distance>m/<planner>/<scenario>_<position>_<index>/eval_result.json`.
The scripts under `socialRPF/example/robot_person_following/visualize_scripts/`
aggregate these JSON files into the bar charts and trend curves shown in the
paper. Each script reads exactly the log directory written by one (or more)
of the `evaluate.sh` runs above:

| Visualize script | Reads from `<repo>/logs/...`                                          | Produced by `evaluate.sh` running                                   | Figure |
|------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------|--------|
| `visualize_results_humans.py`     | `dynamic-1.5m/`, `topography-1.5m/`                       | `config_dynamic_crowd_eval.yaml`, `config_topography_{corridor,doorway,intersection,clutter}_eval.yaml` | Dynamic (H) + Layout (H) bar charts |
| `visualize_results_widths.py`     | `topography-1.5m/`                                         | `config_topography_{corridor,doorway,intersection,clutter}_eval.yaml`                                  | Layout (O) — width / obstacle-count sweep |
| `visualize_results_Fdist.py`      | `dynamic-{1.0,1.5,2.0,2.5}m/` and `topography-{1.0,...}m/` | `config_dynamic_crowd_eval_dynamicF.yaml` and `config_topography_eval_layoutF.yaml` at all four distances | Dynamic (F) + Layout (F) trends vs. following distance |
| `visualize_results_humans_sfm.py` | `dynamic_sfm-1.5m/`                                        | `config_dynamic_crowd_sfm_eval{1,2,3,4}.yaml`                                                          | Same as `humans.py` but with SFM crowd behaviour |
| `visualize_results_all.py`        | `all/` (manually aggregated)                               | All of the above, copied / symlinked into one flat directory        | Cross-experiment summary used in the headline figures |
| `compute_ci_stats.py`             | `all/`                                                     | (same as `_all.py`)                                                 | Mean ± std table for the LaTeX paper |

### Run the visualization

```bash
conda activate follow-bench
cd socialRPF/example/robot_person_following/visualize_scripts

# Run them one at a time, or
python visualize_results_humans.py
python visualize_results_widths.py
python visualize_results_Fdist.py
python visualize_results_humans_sfm.py

# or use the wrapper that runs the *_humans / _widths / _Fdist trio in one shot:
bash run_eval.sh
```

Generated PNGs are written to `<repo>/eval_results/eval_results/`. Override
the location with `FOLLOWBENCH_EVAL_RESULTS=/abs/path` (see the path-conventions
table above).

> **Note on `visualize_results_all.py` / `compute_ci_stats.py`** — these expect a
> single flat `<repo>/logs/all/<planner>/...` tree with every experiment merged
> together. Build it once with rsync / symlinks after the per-experiment
> `evaluate.sh` runs are done, e.g.
> ```bash
> mkdir -p logs/all
> for d in logs/dynamic-1.5m logs/topography-1.5m logs/target_traj-1.5m logs/dynamic_sfm-1.5m; do
>   rsync -a "$d/" logs/all/
> done
> ```

## Evaluation metrics
### Supported
| Metric | Unit | Explanation |
| --- | --- | --- |
| Success Rate $\uparrow$ | - | Runs with no collision |
| Path Length $\downarrow$ | [m] | Robot path length |
| Search Path Length $\downarrow$ | [m] | Robot path length in search process |
| Velocity (avg.) $\downarrow$ | [m/s] | Robot velocity |
| Acceleration (avg.) $\downarrow$ | [m/s²] | Robot acceleration |
| Movement Jerk $\downarrow$ | [m/s³] | Derivation of acceleration |
| Time in personal zone $\uparrow$ | [s] | Time the robot in personal zone of the target |
| Time in private zone $\uparrow$ | [s] | Time the robot in private zone of the surrounding humans |
| Time in search $\uparrow$ | [s] | Time the robot in search process |


## Keep the objects' n-step trajectory
- Add `keep_trail_length`, `keep_traj_length`, `show_trajectory`, and `show_trail` to the `plot{}` section of your YAML file:
  ```yaml
  plot: {show_trajectory: true, show_goal: true, show_trail: true, keep_trail_length: 10, keep_traj_length: 10}

- The pre-generated scenarios under `socialRPF/dynamic_scenarios/config/` and `socialRPF/layout_scenarios/config/` already include this `plot{}` block, so the trail/trajectory rendering works out of the box for any planner that uses them.

## Acknowledgments

This benchmark is built on top of two MIT-licensed open-source projects by
**Han Ruihua**:

- [`hanruihua/RDA-planner`](https://github.com/hanruihua/RDA-planner) — the
  ADMM-based MPC solver bundled in `socialRPF/RDA_planner/`. The package
  retains its original directory name as a visible attribution; the
  upstream copyright notice is preserved verbatim in
  [`socialRPF/LICENSE-RDA-planner`](socialRPF/LICENSE-RDA-planner).
- [`hanruihua/ir-sim`](https://github.com/hanruihua/ir-sim) — the
  lightweight simulator vendored in `ir-sim/`, with light modifications for
  benchmark recording and lidar `build_map`.

We gratefully thank the authors of both projects. Per-component provenance
is documented in [`socialRPF/NOTICE.md`](socialRPF/NOTICE.md).
