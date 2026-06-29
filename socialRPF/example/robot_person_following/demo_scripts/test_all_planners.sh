#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo / smoke-test runner for every `*_diff.py` planner entry point.
#
# Each planner is launched on the same scenario for a small number of steps
# with `--headless` so the script can be used as an end-to-end installation
# check inside a fresh conda env. The exit code of the loop is non-zero when
# any planner fails.
#
# Usage:
#     conda activate <your-env>
#     cd socialRPF/example/robot_person_following
#     bash demo_scripts/run_demo.sh                # run all planners
#     bash demo_scripts/run_demo.sh rda_planner_diff dwa_planner_diff
#     STEPS=200 bash demo_scripts/run_demo.sh      # longer roll-outs
# -----------------------------------------------------------------------------

set -u

# --- Paths -------------------------------------------------------------------

# Directory containing this script -> example/robot_person_following/demo_scripts
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd -- "${EXAMPLE_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/follow_bench_test_logs}"
DYNAMIC_CONFIG_DIR="${DYNAMIC_CONFIG_DIR:-${ROOT}/socialRPF/dynamic_scenarios/config}"

# --- Demo scenario shared by every planner ----------------------------------

SCENARIO="${SCENARIO:-NormalPerpendicularOrcaH10W0}"
POSITION="${POSITION:-back}"
DISTANCE="${DISTANCE:-1.5}"
INDEX="${INDEX:-1}"
STEPS="${STEPS:-50}"  # short roll-out so the loop finishes quickly

# Headless by default; set HEADLESS=0 to keep the matplotlib window.
HEADLESS_FLAG="--headless"
if [[ "${HEADLESS:-1}" == "0" ]]; then
    HEADLESS_FLAG=""
fi

# Servers without a writable home cache need this.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

# --- Planners under test -----------------------------------------------------

ALL_PLANNERS=(
    "rda_planner_diff"
    "rda_traj_planner_diff"
    "rda_dyna_search_planner_diff"
    "dwa_planner_diff"
    "dwa_improved_planner_diff"
    "sfm_planner_diff"
    "bso_hfc_planner_diff"
    "adap_rpf_diff"
    "RL_diff"
)

# Allow running a subset, e.g. `bash run_demo.sh rda_planner_diff sfm_planner_diff`.
if [[ $# -gt 0 ]]; then
    PLANNERS=("$@")
else
    PLANNERS=("${ALL_PLANNERS[@]}")
fi

cd "${EXAMPLE_DIR}"

mkdir -p "${LOG_DIR}"

# --- Run loop ----------------------------------------------------------------

declare -a RESULTS
overall_rc=0

for planner in "${PLANNERS[@]}"; do
    script="${EXAMPLE_DIR}/${planner}.py"
    if [[ ! -f "${script}" ]]; then
        echo "[SKIP] ${planner}: ${script} not found"
        RESULTS+=("SKIP  ${planner}")
        continue
    fi

    echo ""
    echo "============================================================"
    echo " Running ${planner}.py"
    echo "============================================================"
    echo "  scenario : ${SCENARIO}_${POSITION}_${INDEX}"
    echo "  steps    : ${STEPS}"
    echo "  log dir  : ${LOG_DIR}/${planner}"

    set +e
    python "${planner}.py" \
        -c "${DYNAMIC_CONFIG_DIR}" \
        -s "${SCENARIO}" \
        -p "${POSITION}" \
        -d "${DISTANCE}" \
        -m "${STEPS}" \
        -l "${LOG_DIR}" \
        -i "${INDEX}" \
        ${HEADLESS_FLAG}
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
        echo "[PASS] ${planner} (exit ${rc})"
        RESULTS+=("PASS  ${planner}")
    else
        echo "[FAIL] ${planner} (exit ${rc})"
        RESULTS+=("FAIL  ${planner} (exit ${rc})")
        overall_rc=1
    fi
done

# --- Summary -----------------------------------------------------------------

echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
for line in "${RESULTS[@]}"; do
    echo "  ${line}"
done

exit ${overall_rc}
