import matplotlib.pyplot as plt
import numpy as np
import os
import json
from constants import METHODS, COLORS, EVALUATED_METRICS, FOLLOWING_POSITIONS, LOGS_ALL, EVAL_RESULTS_DIR

def load_data(base_dir, method, following_positions, scenarios, trials, evaluated_metrics):
    evaluation_data = {}
    for metric in evaluated_metrics:
        evaluation_data[metric] = []
    evaluation_data["alg_cost_t"] = []
    for trial in trials:
        for scenario in scenarios:
            for following_position in following_positions:
                file_path = os.path.join(
                    base_dir,
                    method,
                    f"{scenario}_{following_position}_{trial}",
                    "eval_result.json"
                )
                detailed_file_path = os.path.join(
                    base_dir,
                    method,
                    f"{scenario}_{following_position}_{trial}",
                    "object_info.json"
                )
                # print(detailed_file_path)
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for metric in evaluated_metrics:
                            if metric in data:
                                evaluation_data[metric].append(data[metric])
                    with open(detailed_file_path, encoding="utf-8") as f:
                        for line in f:
                            object_info = json.loads(line)
                            # print(object_info)
                            if "alg_cost_t" in object_info:
                                evaluation_data["alg_cost_t"].append(object_info['alg_cost_t'])
    return evaluation_data

def plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, following_positions, saved_fname, output_dir):
    evaluated_data = {}
    for method in methods.keys():
        evaluated_data[method] = load_data(base_dir, method, following_positions, scenarios, trials, evaluated_metrics)
        if "alg_cost_t" in evaluated_data[method] and len(evaluated_data[method]['alg_cost_t']) > 0:
            print(f"{method} average alg_cost_t: {np.mean(evaluated_data[method]['alg_cost_t']):.4f}, {np.std(evaluated_data[method]['alg_cost_t']):.4f}")

    rows = 1
    cols = int(np.ceil(len(plotted_metrics) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 3.0))
    axs = axs.flatten()

    for i, (plotted_metric, value) in enumerate(plotted_metrics.items()):
        ax = axs[i]
        ax.set_ylabel(value[-1], fontsize=8, labelpad=0, fontweight='bold')  # keep the unit label close to the axis
        # print(plotted_metric)
        for j, (method, label) in enumerate(methods.items()):
            if plotted_metric in ["ASR", "TVR"]:
                data = np.array(evaluated_data[method][value[0]]) * 100
                # print(max(data))
            elif plotted_metric == "SR":
                data = np.array(evaluated_data[method][value[0]]) * np.array(evaluated_data[method][value[1]]) * 100
            elif plotted_metric in ["PL", "AvgVel", "AvgAcc", "Jerk"]:
                data = np.array(evaluated_data[method][value[0]])
            elif plotted_metric == "TinTPerson" or plotted_metric == "TinPrivate":
                data = np.array(evaluated_data[method][value[0]]) / (np.array(evaluated_data[method][value[1]])*0.1) * 100
            else:
                raise ValueError(f"Unsupported plotted metric: {plotted_metric}")

            if len(data) > 0:
                mean_val = np.mean(data)
                # print(f"{method} {plotted_metric}: {mean_val:.2f}")
                if plotted_metric in ["ASR", "TVR", "SR", "TinTPerson", "TinPrivate"]:
                    ax.bar(j, mean_val, width=0.7, capsize=4,
                       color=colors[method], edgecolor='black', linewidth=1.2,
                       label=label if i == 0 else None)
                    ax.set_ylim(0, 105)
                else:
                    std_val = np.std(data)
                    # print(std_val)
                    ax.bar(j, mean_val, yerr=std_val, width=0.7, capsize=4,
                        color=colors[method], edgecolor='black', linewidth=1.2,
                        label=label if i == 0 else None)

        ax.set_title(plotted_metric, fontsize=10, fontweight='bold')
        ax.set_ylabel(value[-1], fontsize=8)
        ax.set_xticks([])
        if plotted_metric not in ["ASR", "TVR", "SR", "TinTPerson", "TinPrivate"]:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # remove empty subplots
    for k in range(i+1, len(axs)):
        fig.delaxes(axs[k])

    fig.legend(methods.values(), loc='lower center', ncol=len(methods)//rows, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.3, hspace=0.25)  # reduce spacing between subplots
    plt.savefig(os.path.join(output_dir, "{}.png".format(saved_fname)), dpi=400, transparent=True)
    # plt.show()





if __name__ == "__main__":
    """
    1. Visualize each following direction separately.
    2. Compare all methods in the same figure.
    3. Average across all scenarios.
    """
    base_dir = LOGS_ALL
    following_positions = FOLLOWING_POSITIONS
    methods = METHODS
    colors = COLORS
    evaluated_metrics = EVALUATED_METRICS

    ### Target Trajectory ###
    scenarios = [
        "two_triangles",
        "eight",
        "two_squares",
        "lShape60",
        "lShape45",
        "lShape30",
        "uTurn",
        "backAndForth",
    ]
    trials = [i for i in range(1)]
    plotted_metrics = {
        # "ASR": ["obstacle_avoidance_success", "%"],
        "SR": ["obstacle_avoidance_success", "search_success", "%"],
        # "TVR": ["target_visibility_ratio", "%"],
        "TinTPerson": ["time_in_target_personal_zone", "max_steps", "%"],
        # "TinPrivate": ["time_in_private_zone", "max_steps", "%"],
        # "SPLs": ["search_path_length", ""],
        "PL": ["path_length", "m"],
        "AvgVel": ["avg_velocity", "m/s"],
        "AvgAcc": ["avg_acceleration", "m/s$^2$"],
        "Jerk": ["avg_jerk", "m/s$^3$"],
    }
    print("=== Evaluating Target Traj ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, following_positions, "expTargetTraj", EVAL_RESULTS_DIR)


    base_dir = LOGS_ALL
    ### Dynamic Crowds ###
    scenarios = [
        "NormalCrowdOrcaH5W0",
        "NormalCrowdOrcaH10W0",
        "NormalCrowdOrcaH15W0",
        "NormalCrowdOrcaH20W0",
        "NormalCrowdOrcaH25W0",
        "NormalCrowdOrcaH30W0",
        # "NormalCrowdOrcaH35W0",
        # "NormalCrowdOrcaH40W0",
        "NormalCircularOrcaH5W0",
        "NormalCircularOrcaH10W0",
        "NormalCircularOrcaH15W0",
        "NormalCircularOrcaH20W0",
        "NormalCircularOrcaH25W0",
        "NormalCircularOrcaH30W0",
        # "NormalCircularOrcaH35W0",
        # "NormalCircularOrcaH40W0",
        "NormalParallelOrcaH5W0",
        "NormalParallelOrcaH10W0",
        "NormalParallelOrcaH15W0",
        "NormalParallelOrcaH20W0",
        "NormalParallelOrcaH25W0",
        "NormalParallelOrcaH30W0",
        # "NormalParallelOrcaH35W0",
        # "NormalParallelOrcaH40W0",
        "NormalPerpendicularOrcaH5W0",
        "NormalPerpendicularOrcaH10W0",
        "NormalPerpendicularOrcaH15W0",
        "NormalPerpendicularOrcaH20W0",
        "NormalPerpendicularOrcaH25W0",
        "NormalPerpendicularOrcaH30W0",
        # "NormalPerpendicularOrcaH35W0",
        # "NormalPerpendicularOrcaH40W0"
        ]

    trials = [i for i in range(1, 101)]
    plotted_metrics = {
        "SR": ["obstacle_avoidance_success", "search_success", "%"],
        "ASR": ["obstacle_avoidance_success", "%"],
        "TVR": ["target_visibility_ratio", "%"],
        "TinTPerson": ["time_in_target_personal_zone", "max_steps", "%"],
        "TinPrivate": ["time_in_human_private_zone", "max_steps", "%"],
        # "SPLs": ["search_path_length", ""],
        # "PL": ["path_length", "m"],
        # "AvgVel": ["avg_velocity", "m/s"],
        # "AvgAcc": ["avg_acceleration", "m/s$^2$"],
        "Jerk": ["avg_jerk", "m/s$^3$"],
    }
    print("=== Evaluating Dynamic ===")
    dynamic_scenarios = list(scenarios)
    dynamic_trials = list(trials)
    plot_results(base_dir, methods, dynamic_scenarios, dynamic_trials, evaluated_metrics, plotted_metrics, colors, following_positions, "expDynamic", EVAL_RESULTS_DIR)

    base_dir = LOGS_ALL
    ### Topographies ###
    scenarios = [
        "ClutterOrcaObs30H10",
        "ClutterOrcaObs30H20",
        "ClutterOrcaObs30H30",
        "ClutterOrcaObs30H40",
        "ClutterOrcaObs10H30",
        "ClutterOrcaObs20H30",
        "ClutterOrcaObs40H30",

        "CorridorOrcaH4W5.6",
        "CorridorOrcaH12W5.6",
        "CorridorOrcaH20W5.6",
        "CorridorOrcaH28W5.6",
        "CorridorOrcaH20W5.0",
        "CorridorOrcaH20W6.2",
        "CorridorOrcaH20W6.8",

        "DoorwayOrcaH2W2.8",
        "DoorwayOrcaH6W2.8",
        "DoorwayOrcaH10W2.8",
        "DoorwayOrcaH14W2.8",
        "DoorwayOrcaH10W2.5",
        "DoorwayOrcaH10W3.1",
        "DoorwayOrcaH10W3.4",
        
        "IntersectionOrcaH2W4.8",
        "IntersectionOrcaH10W4.8",
        "IntersectionOrcaH18W4.8",
        "IntersectionOrcaH26W4.8",
        "IntersectionOrcaH18W4.2",
        "IntersectionOrcaH18W5.4",
        "IntersectionOrcaH18W6.0",
        ]
    trials = [i for i in range(1, 101)]
    plotted_metrics = {
        "SR": ["obstacle_avoidance_success", "search_success", "%"],
        "ASR": ["obstacle_avoidance_success", "%"],
        "TVR": ["target_visibility_ratio", "%"],
        "TinTPerson": ["time_in_target_personal_zone", "max_steps", "%"],
        "TinPrivate": ["time_in_human_private_zone", "max_steps", "%"],
        # "SPLs": ["search_path_length", ""],
        # "PL": ["path_length", "m"],
        # "AvgVel": ["avg_velocity", "m/s"],
        # "AvgAcc": ["avg_acceleration", "m/s$^2$"],
        "Jerk": ["avg_jerk", "m/s$^3$"],
    }
    print("=== Evaluating Layout ===")
    layout_scenarios = list(scenarios)
    layout_trials = list(trials)
    plot_results(base_dir, methods, layout_scenarios, layout_trials, evaluated_metrics, plotted_metrics, colors, following_positions, "expLayout", EVAL_RESULTS_DIR)

    # ------------------------------------------------------------------
    # Combined planner timing stats (Dynamic + Layout)
    # ------------------------------------------------------------------
    combined_scenarios = dynamic_scenarios + layout_scenarios
    combined_trials = sorted(set(dynamic_trials) | set(layout_trials))
    print("\n" + "=" * 80)
    print("  Planner Timing: Dynamic + Layout combined (alg_cost_t)")
    print("=" * 80)
    print(f"  {'Method':<35s}  {'Mean (s)':>10s}  {'Std (s)':>10s}  {'Mean (ms)':>10s}  {'Std (ms)':>10s}  {'N':>8s}")
    print("-" * 95)
    for method, label in methods.items():
        timing_data = load_data(base_dir, method, following_positions, combined_scenarios, combined_trials, evaluated_metrics)
        costs = timing_data.get("alg_cost_t", [])
        if len(costs) == 0:
            print(f"  {label:<35s}  {'--':>10s}  {'--':>10s}  {'--':>10s}  {'--':>10s}  {'0':>8s}")
            continue
        arr = np.array(costs)
        mean_s = np.mean(arr)
        std_s = np.std(arr)
        print(f"  {label:<35s}  {mean_s:10.4f}  {std_s:10.4f}  {mean_s*1000:10.2f}  {std_s*1000:10.2f}  {len(arr):8d}")
    print("=" * 80)