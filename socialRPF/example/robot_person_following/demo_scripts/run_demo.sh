#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Single-shot smoke test for one of the *_diff.py planners.
#
# This script auto-detects the repository root, so you do not need to edit any
# absolute path before running it. Override LOG_DIR / DYNAMIC_CONFIG_DIR /
# STATIC_CONFIG_DIR via environment variables if you want logs / scenarios to
# live elsewhere.
#
# Available planner CLI arguments (mirrored from each *_diff.py):
#   -c, --config_path   directory containing scenario yaml files (required)
#   -s, --scenario      scenario name, e.g. NormalCircularOrcaH15W0
#   -p, --position      back / left_side / right_side
#   -d, --distance      following distance (m)
#   -m, --min_steps     min steps for the target to finish the trajectory
#   -i, --index         trial index inside the (scenario, position) bucket
#   -l, --log_path      evaluation log directory
#   -t, --traj_predictor   trajectory predictor: cv | cvkf | sgan
#   -v, --visualize     turn on rendering
# ----------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FOLLOWBENCH_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
DYNAMIC_CONFIG_DIR="${DYNAMIC_CONFIG_DIR:-$ROOT/socialRPF/dynamic_scenarios/config}"
STATIC_CONFIG_DIR="${STATIC_CONFIG_DIR:-$ROOT/socialRPF/layout_scenarios/config}"

cd "$ROOT/socialRPF/example/robot_person_following"

# ---- dynamic crowd demo ---------------------------------------------------
# Available scenarios (examples): NormalCircularOrcaH15W0, NormalCrowdOrcaH10W0,
# NormalParallelOrcaH10W0, NormalPerpendicularOrcaH10W0
python rda_planner_diff.py -c "$DYNAMIC_CONFIG_DIR" -s NormalPerpendicularOrcaH10W0 -p back -d 1.5 -m 800 -l "$LOG_DIR" -i 1

# python sfm_planner_diff.py -c "$DYNAMIC_CONFIG_DIR" -s NormalPerpendicularOrcaH10W0 -p left_side -d 1.5 -m 800 -l "$LOG_DIR" -i 1
# python adap_rpf_diff.py -c "$DYNAMIC_CONFIG_DIR" -s NormalPerpendicularOrcaH10W0 -p back -d 1.5 -m 200 -l "$LOG_DIR" -i 1 --headless

# ---- static layout demo ---------------------------------------------------
# Available scenarios (examples): ClutterOrcaObs30H10, CorridorOrcaH20W5.6,
# DoorwayOrcaH10W2.8, IntersectionOrcaH18W4.8
# python rda_planner_diff.py -c "$STATIC_CONFIG_DIR" -s IntersectionOrcaH18W4.8 -p left_side -d 1.5 -m 800 -l "$LOG_DIR" -i 1
