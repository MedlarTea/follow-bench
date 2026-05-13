#!/usr/bin/env bash
# run_eval.sh — Run all evaluation visualization scripts in one shot
# Output figures are saved to EVAL_RESULTS_DIR (defined in constants.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=python3
LOG_FILE="$SCRIPT_DIR/run_eval.log"
PASS=0
FAIL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPTS=(
   "visualize_results_all.py"
    "visualize_results_Fdist.py"
    "visualize_results_humans.py"
    "visualize_results_widths.py"
)

echo "=================================================="
echo " Follow-Bench Evaluation  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo "Log: $LOG_FILE"
echo ""
> "$LOG_FILE"

for script in "${SCRIPTS[@]}"; do
    printf "Running %-40s ... " "$script"
    if $PYTHON "$script" >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}OK${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAILED${NC}"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "--------------------------------------------------"
echo -e "Result: ${GREEN}${PASS} passed${NC}  ${RED}${FAIL} failed${NC}"

# List the generated figures
EVAL_RESULTS_DIR=$($PYTHON -c "from constants import EVAL_RESULTS_DIR; print(EVAL_RESULTS_DIR)")
echo ""
echo -e "${YELLOW}Output:${NC} $EVAL_RESULTS_DIR"
if ls "$EVAL_RESULTS_DIR"/*.png 2>/dev/null | head -20; then
    :
else
    echo "  (no PNG files found)"
fi

[ "$FAIL" -eq 0 ]
