import os
from pathlib import Path

# ─── Repository root auto-detection ───────────────────────────────────────────
# This file lives at:
#   <REPO_ROOT>/socialRPF/example/robot_person_following/visualize_scripts/constants.py
# So the repo root is four directories up.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# ─── Log directories ──────────────────────────────────────────────────────────
# By default, evaluation logs are written under ``<REPO_ROOT>/logs/`` and the
# generated figures under ``<REPO_ROOT>/eval_results/``. Override these via
# environment variables when you store results elsewhere:
#
#   export FOLLOWBENCH_LOG_DIR=/abs/path/to/logs
#   export FOLLOWBENCH_EVAL_RESULTS=/abs/path/to/figures
LOG_BASE          = os.environ.get("FOLLOWBENCH_LOG_DIR",     str(_REPO_ROOT / "logs"))
EVAL_RESULTS_BASE = os.environ.get("FOLLOWBENCH_EVAL_RESULTS", str(_REPO_ROOT / "eval_results"))

# Layout matches what `evaluate_scripts/evaluate.sh` writes (it appends
# `-<distance>m` to whatever `log_dir` is declared in the eval yaml). Each
# entry below corresponds to one or more eval yaml under `evaluate_scripts/`:
#
#   evaluate.sh writes ...          | constant used by ...
#   --------------------------------+--------------------------------------------
#   logs/dynamic-<D>m/              | LOGS_DYNAMIC          (Fdist sweep, prefix)
#   logs/dynamic-1.5m/              | LOGS_DYNAMIC_1_5M     (Dynamic H sweep, fixed 1.5m)
#   logs/topography-<D>m/           | LOGS_STATIC           (Layout F sweep, prefix)
#   logs/topography-1.5m/           | LOGS_STATIC_1_5M      (Layout H/O sweep, fixed 1.5m)
#   logs/dynamic_sfm-1.5m/          | LOGS_SFM              (SFM crowd eval)
#   <user-aggregated dir>           | LOGS_ALL              (cross-experiment plots)
LOGS_DYNAMIC      = os.path.join(LOG_BASE, "dynamic")           # prefix: real dirs are dynamic-{1.0,1.5,2.0,2.5}m
LOGS_STATIC       = os.path.join(LOG_BASE, "topography")        # prefix: real dirs are topography-{1.0,1.5,2.0,2.5}m
LOGS_DYNAMIC_1_5M = os.path.join(LOG_BASE, "dynamic-1.5m")      # dynamic crowd at fixed F=1.5m
LOGS_STATIC_1_5M  = os.path.join(LOG_BASE, "topography-1.5m")   # static layout at fixed F=1.5m
LOGS_SFM          = os.path.join(LOG_BASE, "dynamic_sfm-1.5m")  # SFM crowd at fixed F=1.5m
LOGS_ALL          = os.path.join(LOG_BASE, "all")               # combined / aggregated logs (manual)

# ─── Visualization output directory ───────────────────────────────────────────

EVAL_RESULTS_DIR = os.path.join(EVAL_RESULTS_BASE, "eval_results")
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# ─── Methods ──────────────────────────────────────────────────────────────────

METHODS = {
    "rda_planner_diff":             "MPC",
    "rda_traj_planner_diff":        "MPC w/ Traj.",
    "rda_dyna_search_planner_diff": "MPC w/ DS.",
    "sfm_planner_diff":             "SFM",
    "dwa_planner_diff":             "DWA",
    "dwa_improved_planner_diff":    "DWA w/ Traj.",
    "bso_hfc_planner_diff":         "BSO-HFC",
    "RL-based1_planner_diff":       "RL-based",
    # "RL-based2_planner_diff":       "RL-based2",
}

COLORS = {
    "rda_planner_diff":             "#0780cf",
    "rda_traj_planner_diff":        "#765005",
    "rda_dyna_search_planner_diff": "#fa6d1d",
    "sfm_planner_diff":             "#0e2c82",
    "dwa_planner_diff":             "#b6b51f",
    "dwa_improved_planner_diff":    "#da1f18",
    "bso_hfc_planner_diff":         "#13a8a8",
    "RL-based1_planner_diff":       "#9b59b6",
    # "RL-based2_planner_diff":       "#e67e22",
}

# ─── Shared experiment config ─────────────────────────────────────────────────

FOLLOWING_POSITIONS = ["back", "left_side"]

LINE_STYLES = {
    "back":      "-",
    "left_side": "--",
}

EVALUATED_METRICS = [
    "max_steps",
    "max_search_steps",
    "obstacle_avoidance_success",
    "target_visibility_ratio",
    "search_success",
    "search_path_length",
    "avg_robot_target_dist_no_radius",
    "path_length",
    "avg_velocity",
    "avg_acceleration",
    "avg_jerk",
    "time_in_target_personal_zone",
    "time_in_human_private_zone",
    "time_in_target_search",
    "total_time",
]
