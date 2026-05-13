"""Aggregate per-scenario `simulation_steps.json` entries into per-prefix
median+1 statistics. Used to derive the `*_min_steps.json` baselines under
``socialRPF/layout_scenarios/config/01_min_step/``.
"""
import argparse
import json
import os

import numpy as np


def next_after_median(arr):
    arr = np.asarray(arr)  # convert to a numpy array
    med = np.median(arr)   # median
    greater = arr[arr > med]  # values greater than the median
    sorted_greater = np.sort(greater)  # sort them
    if greater.size > 50:
        return sorted_greater[50]  # return the 51st (index 50)
    else:
        return int(med)


def aggregate(input_path: str) -> dict[str, int]:
    with open(input_path, "r") as f:
        data = json.load(f)

    grouped: dict[str, list] = {}
    for key in sorted(data.keys()):
        scenario = key.split("_")[0]
        grouped.setdefault(scenario, []).append(data[key])

    return {scene: int(next_after_median(values)) for scene, values in grouped.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=str,
        help="Path to a simulation_steps.json file produced by the scenario "
             "generation toolchain.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional output JSON path; if omitted the aggregated table is "
             "printed to stdout.",
    )
    args = parser.parse_args()

    result = aggregate(args.input)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"wrote {len(result)} entries to {args.output}")
    else:
        for scene, value in result.items():
            print(f"{scene}: {value}")
