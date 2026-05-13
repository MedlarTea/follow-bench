"""
Compute mean +/- std for key continuous metrics (TVR, TinTPerson, Jerk)
across two representative scenarios (Dynamic H=20, Layout: Doorway H=10 W=2.8)
for MPC-based planners. Results are formatted for the LaTeX table (Tab. ciStats).

Usage:
    conda activate followBench
    python compute_ci_stats.py
"""

import os
import json
import numpy as np
from constants import LOGS_ALL, EVALUATED_METRICS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = LOGS_ALL

# Only MPC-based planners
METHODS = {
    "rda_planner_diff":             "MPC",
    "rda_traj_planner_diff":        "MPC w/ Traj.",
    "rda_dyna_search_planner_diff": "MPC w/ DS.",
}

# Side-following at 1.5 m
FOLLOWING_POSITION = "back"

TRIALS = list(range(1, 101))

# Two representative scenario groups
SCENARIO_GROUPS = {
    "Dynamic: Perpendicular Crossing (N_human=20)": [
        "NormalPerpendicularOrcaH20W0",
    ],
    "Layout: Cluttered Space (20 obstacles, 30 humans)": [
        "ClutterOrcaObs20H30",
    ],
}

# ---------------------------------------------------------------------------
# Data loading (mirrors visualize_results_all.py)
# ---------------------------------------------------------------------------

def load_trials(base_dir, method, scenarios, trials, position):
    """Load per-trial metrics and return raw arrays."""
    records = []
    missing = 0
    for trial in trials:
        for scenario in scenarios:
            trial_dir = f"{scenario}_{position}_{trial}"
            fpath = os.path.join(base_dir, method, trial_dir, "eval_result.json")
            if not os.path.exists(fpath):
                missing += 1
                continue
            with open(fpath, "r") as f:
                data = json.load(f)
            records.append(data)
    if missing > 0:
        print(f"  [warn] {method}: {missing} trial files not found")
    return records


def compute_metrics(records):
    """Compute SR (%), TVR (%), TinTPerson (%), Jerk (m/s^3) from raw trial records."""
    sr_vals = []
    tvr_vals = []
    tin_vals = []
    jerk_vals = []

    for d in records:
        # SR: obstacle_avoidance_success AND search_success (binary -> 0/100)
        oa = d.get("obstacle_avoidance_success", None)
        ss = d.get("search_success", None)
        if oa is not None and ss is not None:
            sr_vals.append(float(oa) * float(ss) * 100.0)

        # TVR: target_visibility_ratio is already a fraction [0, 1]
        tvr = d.get("target_visibility_ratio", None)
        if tvr is not None:
            tvr_vals.append(tvr * 100.0)

        # TinTPerson: time_in_target_personal_zone / total_sim_time * 100
        t_personal = d.get("time_in_target_personal_zone", None)
        max_steps = d.get("max_steps", None)
        if t_personal is not None and max_steps is not None and max_steps > 0:
            tin_vals.append(t_personal / (max_steps * 0.1) * 100.0)

        # Jerk: avg_jerk (m/s^3)
        jerk = d.get("avg_jerk", None)
        if jerk is not None:
            jerk_vals.append(jerk)

    return {
        "SR":         (np.array(sr_vals),   "%"),
        "TVR":        (np.array(tvr_vals),  "%"),
        "TinTPerson": (np.array(tin_vals),  "%"),
        "Jerk":       (np.array(jerk_vals), "m/s^3"),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "-" * 80

    # Collect results for LaTeX output
    latex_rows = {}  # (scenario_group, method) -> formatted strings

    for group_name, scenarios in SCENARIO_GROUPS.items():
        print(f"\n{sep}")
        print(f"  Scenario: {group_name}")
        print(f"  Sub-scenarios: {scenarios}")
        print(f"  Following: {FOLLOWING_POSITION}, Trials: {TRIALS[0]}-{TRIALS[-1]}")
        print(sep)

        for method_key, method_label in METHODS.items():
            records = load_trials(BASE_DIR, method_key, scenarios,
                                  TRIALS, FOLLOWING_POSITION)
            if len(records) == 0:
                print(f"  {method_label:20s}  ** NO DATA **")
                latex_rows[(group_name, method_label)] = ("--", "--", "--")
                continue

            metrics = compute_metrics(records)
            parts = []
            for metric_name in ["SR", "TVR", "TinTPerson", "Jerk"]:
                arr, unit = metrics[metric_name]
                if len(arr) == 0:
                    parts.append("--")
                    print(f"  {method_label:20s}  {metric_name:12s}  ** NO DATA **")
                    continue

                mean = np.mean(arr)
                std  = np.std(arr)
                n    = len(arr)

                if metric_name == "Jerk":
                    fmt = f"{mean:.2f} +/- {std:.2f}"
                else:
                    fmt = f"{mean:.1f} +/- {std:.1f}"
                parts.append(fmt)
                print(f"  {method_label:20s}  {metric_name:12s}  "
                      f"{fmt:>20s}  (n={n})")

            latex_rows[(group_name, method_label)] = tuple(parts)

    # -----------------------------------------------------------------------
    # Print LaTeX-ready table rows
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  LaTeX table rows (copy-paste into benchExp.tex Tab. ciStats)")
    print(f"{'=' * 80}\n")

    for group_name, scenarios in SCENARIO_GROUPS.items():
        print(f"% --- {group_name} ---")
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        print("\\midrule")
        for method_label in METHODS.values():
            sr, tvr, tin, jerk = latex_rows[(group_name, method_label)]
            # Convert +/- to $\pm$ with \scriptsize for std
            def fmt_latex(s):
                if "+/-" not in s:
                    return s
                mean, std = s.split(" +/- ")
                return f"{mean} {{\\scriptsize $\\pm$ {std}}}"
            print(f"{method_label:20s} & {fmt_latex(sr)} & {fmt_latex(tvr)} & {fmt_latex(tin)} & {fmt_latex(jerk)} \\\\")
        print()


if __name__ == "__main__":
    main()
