#!/bin/bash
# -----------------------------------------------------------------------------
# Run a full evaluation sweep described by a yaml file under this directory.
#
# Usage:
#     bash evaluate.sh                              # use the default yaml + d
#     bash evaluate.sh config_dynamic_crowd_eval.yaml
#     bash evaluate.sh config_dynamic_crowd_eval.yaml 1.2
#     bash evaluate.sh config_dynamic_crowd_eval.yaml 1.2 --headless
#
# Path conventions (no need to edit the yaml files):
#   * The repository root is auto-detected (FOLLOWBENCH_ROOT overrides it).
#   * `config_dir` and `log_dir` inside the yaml are interpreted as paths
#     relative to that root unless they are already absolute.
#   * Override the log location with `FOLLOWBENCH_LOG_DIR=/abs/path bash ...`
#     to redirect the entire log_dir tree to an external disk.
# -----------------------------------------------------------------------------

# Default configuration
DEFAULT_CONFIG="config_topography_corridor_eval.yaml"
DEFAULT_D=1.5
HEADLESS_FLAG=""

# Auto-detect repo root: this file lives at
#   <REPO_ROOT>/socialRPF/example/robot_person_following/evaluate_scripts/evaluate.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FOLLOWBENCH_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"

# Resolve a possibly-relative path against the repo root.
resolve_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "${REPO_ROOT}/${p}"
    fi
}

# ==============================
# Argument parsing
# ==============================

POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --headless)
            HEADLESS_FLAG="--headless"
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [ $# -eq 0 ]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
    D_VALUE="$DEFAULT_D"
    echo "Using default config file: $CONFIG_FILE"
    echo "Using default d value: $D_VALUE"
elif [ $# -eq 1 ]; then
    CONFIG_FILE="$1"
    D_VALUE="$DEFAULT_D"
    echo "Using config file: $CONFIG_FILE"
    echo "Using default d value: $D_VALUE"
elif [ $# -eq 2 ]; then
    CONFIG_FILE="$1"
    D_VALUE="$2"
    echo "Using config file: $CONFIG_FILE"
    echo "Using d value: $D_VALUE"
else
    echo "Usage: $0 [config_file_path] [d_value] [--headless]"
    echo "Example: $0"
    echo "Example: $0 config.yaml"
    echo "Example: $0 config.yaml 1.2"
    echo "Example: $0 config.yaml 1.2 --headless"
    exit 1
fi

if [[ -n "$HEADLESS_FLAG" ]]; then
    echo "Headless mode enabled"
fi

# ==============================
# Read config
# ==============================

log_dir_raw=$(yq '.log_dir' "$CONFIG_FILE")
config_dir_raw=$(yq '.config_dir' "$CONFIG_FILE")

# Allow FOLLOWBENCH_LOG_DIR to redirect logs (e.g. to an external SSD)
if [[ -n "${FOLLOWBENCH_LOG_DIR:-}" ]]; then
    log_dir_raw="${FOLLOWBENCH_LOG_DIR}/$(basename "$log_dir_raw")"
fi

log_dir="$(resolve_path "$log_dir_raw")"
config_dir="$(resolve_path "$config_dir_raw")"

echo "  REPO_ROOT  : $REPO_ROOT"
echo "  config_dir : $config_dir"
echo "  log_dir    : $log_dir"

readarray -t baselines < <(yq '.baselines[]' "$CONFIG_FILE")
readarray -t p_values < <(yq '.p_values[]' "$CONFIG_FILE")

# Note: n_trials is a single value (not an array)
n_trials=$(yq '.n_trials' "$CONFIG_FILE")

echo "n_trials: $n_trials"

# ==============================
# Main loop
# ==============================

for baseline in "${baselines[@]}"; do

  while IFS="=" read -r scene m_val; do
    scene=$(echo "$scene" | xargs)
    m_val=$(echo "$m_val" | xargs)

    if [[ -z "$m_val" ]]; then
      echo "[Warning] Scene '$scene' has no m value, skipping..."
      continue
    fi

    for p in "${p_values[@]}"; do
      for ((i=1; i<n_trials+1; i++)); do

        echo ""
        echo "Running: [$baseline $scene $p $i d=$D_VALUE]"

        basename_no_ext="${baseline%.*}"

        # Append the following distance (e.g. -1.5m) to the log dir name
        log_dir_with_d="${log_dir}-${D_VALUE}m"

        python ../"$baseline" \
          -s "$scene" \
          -p "$p" \
          -d "$D_VALUE" \
          -m "$m_val" \
          -i "$i" \
          -l "$log_dir_with_d/$basename_no_ext" \
          -c "$config_dir" \
          $HEADLESS_FLAG

      done
    done

  done < <(yq '.scenarios | to_entries | .[] | "\(.key)=\(.value)"' "$CONFIG_FILE")

done
