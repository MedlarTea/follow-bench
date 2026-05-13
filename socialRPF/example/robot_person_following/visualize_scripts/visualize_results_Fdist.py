import matplotlib.pyplot as plt
import numpy as np
import os
import json
from constants import METHODS, COLORS, EVALUATED_METRICS, FOLLOWING_POSITIONS, LINE_STYLES, LOGS_DYNAMIC, LOGS_STATIC, EVAL_RESULTS_DIR

def load_data(base_dir, method, human_nums, following_dists, plotted_following_dists, following_positions, scenarios, trials, evaluated_metrics):
    # initialize a dictionary to hold evaluation data
    evaluation_data = {}
    for following_position in following_positions:
        evaluation_data[following_position] = {}
        for plotted_following_dist in plotted_following_dists:
            evaluation_data[following_position][plotted_following_dist] = {}
            for metric in evaluated_metrics:
                evaluation_data[following_position][plotted_following_dist][metric] = []

    # print(method)
    # print(following_positions)
    # print(plotted_following_dists)
    # print(human_nums)
    # print(scenarios)
    for following_position in following_positions:
        for i, plotted_following_dist in enumerate(plotted_following_dists):
            real_base_dir = base_dir + "-{}m".format(plotted_following_dist)
            for scenario in scenarios:
                # for key, value in human_nums.items():
                #     if key in scenario:
                #         real_human_nums = value
                # if "H{}".format(value) in scenario:
                for trial in trials:
                    file_path = os.path.join(
                        real_base_dir,
                        method,
                        f"{scenario}_{following_position}_{trial}",
                        "eval_result.json"
                    )
                    # print(file_path)
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                    else:
                        # print(file_path)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            for metric in evaluated_metrics:
                                if metric in data:
                                    evaluation_data[following_position][plotted_following_dist][metric].append(data[metric])
                                    if metric == "target_visibility_ratio" and data[metric] > 1.0:
                                        print(f"Warning: target_visibility_ratio > 1.0 in {file_path}")
    return evaluation_data

def plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_dists, plotted_following_dists, following_positions, saved_fname, output_dir):
    evaluated_data = {}
    # method -> following_position -> human_num -> metric -> values
    for method in methods.keys():
        evaluated_data[method] = load_data(base_dir, method, human_nums, following_dists, plotted_following_dists, following_positions, scenarios, trials, evaluated_metrics)

    rows = 1
    cols = int(np.ceil(len(plotted_metrics) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(20, 5))
    axs = axs.flatten()

    for i, (plotted_metric, value) in enumerate(plotted_metrics.items()):
        ax = axs[i]
        ax.set_ylabel(value[-1], fontsize=8, labelpad=0, fontweight='bold')  # keep the unit label close to the axis
        # print(plotted_metric)
        for method, label in methods.items():
            for k, following_position in enumerate(following_positions):
                
                plotted_x = []
                plotted_values = []
                for following_dist in plotted_following_dists:
                    if plotted_metric in ["ASR", "TVR"]:
                        data = np.array(evaluated_data[method][following_position][following_dist][value[0]]) * 100
                    elif plotted_metric == "SR":
                        data = np.array(evaluated_data[method][following_position][following_dist][value[0]]) * np.array(evaluated_data[method][following_position][following_dist][value[1]]) * 100
                    elif plotted_metric in ["PL", "AvgVel", "AvgAcc", "Jerk"]:
                        data = np.array(evaluated_data[method][following_position][following_dist][value[0]])
                    elif plotted_metric == "TinTPerson" or plotted_metric == "TinPrivate":
                        data = np.array(evaluated_data[method][following_position][following_dist][value[0]]) / (np.array(evaluated_data[method][following_position][following_dist][value[1]])*0.1) * 100
                    else:
                        raise ValueError(f"Unsupported plotted metric: {plotted_metric}")
                    if len(data) > 0:
                        plotted_x.append(following_dist)
                        plotted_values.append(np.mean(data))
                        # if method == "rda_planner_diff" or method == "rda_traj_planner_diff":
                        #     print(f"Method: {method}, Position: {following_position}, Humans: {human_num}, Metric: {plotted_metric}, Value: {np.mean(data):.2f}")

                if len(plotted_values) > 0:
                    if plotted_metric in ["ASR", "TVR", "SR", "TinTPerson", "TinPrivate"]:
                        ax.plot(plotted_x, plotted_values, marker='o', color=colors[method], linestyle=line_styles[following_position], label=f"{label}-{following_position}")
                        ax.set_ylim(0, 105)
                    else:
                        ax.plot(plotted_x, plotted_values, marker='o', color=colors[method], linestyle=line_styles[following_position], label=f"{label}-{following_position}")
                        ax.set_ylim(0, 80)

        ax.set_title(plotted_metric, fontsize=14, fontweight='bold')
        ax.set_ylabel(value[-1], fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        # ax.set_xticks([])  
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 

    # remove empty subplots
    for k in range(i+1, len(axs)):
        fig.delaxes(axs[k])
    
    method_labels = [f"{label} ({following_position})" for label in methods.values() for following_position in following_positions]

    fig.legend(method_labels, loc='lower center', ncol=len(methods)//rows, bbox_to_anchor=(0.5, -0.01), fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.3, hspace=0.0)  # reduce spacing between subplots
    plt.savefig(os.path.join(output_dir, "{}.png".format(saved_fname)), dpi=500, transparent=True)
    # plt.show()
    plt.close()





if __name__ == "__main__":
    """
    1. Visualize each following direction separately.
    2. Compare all methods in the same figure.
    3. Average across all scenarios.
    """
    base_dir = LOGS_DYNAMIC
    following_positions = FOLLOWING_POSITIONS
    methods = METHODS
    colors = COLORS
    line_styles = LINE_STYLES
    evaluated_metrics = EVALUATED_METRICS


    ### Dynamic Crowds (ORCA) ###
    scenarios = [
        # "NormalCrowdOrcaH5W0",
        # "NormalCrowdOrcaH10W0",
        # "NormalCrowdOrcaH15W0",
        "NormalCrowdOrcaH20W0",
        # "NormalCrowdOrcaH25W0",
        # "NormalCrowdOrcaH30W0",

        # "NormalCircularOrcaH5W0",
        # "NormalCircularOrcaH10W0",
        # "NormalCircularOrcaH15W0",
        "NormalCircularOrcaH20W0",
        # "NormalCircularOrcaH25W0",
        # "NormalCircularOrcaH30W0",

        # "NormalParallelOrcaH5W0",
        # "NormalParallelOrcaH10W0",
        # "NormalParallelOrcaH15W0",
        "NormalParallelOrcaH20W0",
        # "NormalParallelOrcaH25W0",
        # "NormalParallelOrcaH30W0",

        # "NormalPerpendicularOrcaH5W0",
        # "NormalPerpendicularOrcaH10W0",
        # "NormalPerpendicularOrcaH15W0",
        "NormalPerpendicularOrcaH20W0",
        # "NormalPerpendicularOrcaH25W0",
        # "NormalPerpendicularOrcaH30W0",
        ]
    
    human_nums = {
        "NormalCrowdOrca": [20],
        "NormalCircularOrca": [20],
        "NormalParallelOrca": [20],
        "NormalPerpendicularOrca": [20],
    }

    following_distances = {
        "NormalCrowdOrca": [1.0, 1.5, 2.0, 2.5],
        "NormalCircularOrca": [1.0, 1.5, 2.0, 2.5],
        "NormalParallelOrca": [1.0, 1.5, 2.0, 2.5],
        "NormalPerpendicularOrca": [1.0, 1.5, 2.0, 2.5],
    }

    plotted_following_distances = [1.0, 1.5, 2.0, 2.5]

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
    print("=== Evaluating Dynamic Crowd Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expDynamicFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "NormalCrowdOrcaH5W0",
        # "NormalCrowdOrcaH10W0",
        # "NormalCrowdOrcaH15W0",
        "NormalCrowdOrcaH20W0",
        # "NormalCrowdOrcaH25W0",
        # "NormalCrowdOrcaH30W0",
        ]
    print("=== Evaluating Crowd Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expDynamicCrowdFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "NormalCircularOrcaH5W0",
        # "NormalCircularOrcaH10W0",
        # "NormalCircularOrcaH15W0",
        "NormalCircularOrcaH20W0",
        # "NormalCircularOrcaH25W0",
        # "NormalCircularOrcaH30W0",
        ]
    print("=== Evaluating Circular Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expDynamicCircularFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "NormalParallelOrcaH5W0",
        # "NormalParallelOrcaH10W0",
        # "NormalParallelOrcaH15W0",
        "NormalParallelOrcaH20W0",
        # "NormalParallelOrcaH25W0",
        # "NormalParallelOrcaH30W0",
        ]
    print("=== Evaluating Parallel Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expDynamicParallelFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "NormalPerpendicularOrcaH5W0",
        # "NormalPerpendicularOrcaH10W0",
        # "NormalPerpendicularOrcaH15W0",
        "NormalPerpendicularOrcaH20W0",
        # "NormalPerpendicularOrcaH25W0",
        # "NormalPerpendicularOrcaH30W0",
        ]
    print("=== Evaluating Perpendicular Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expDynamicPerpendicularFDist", EVAL_RESULTS_DIR)

    ### Topographies ###
    base_dir = LOGS_STATIC
    human_nums = {
        "ClutterOrcaObs": [30],
        "CorridorOrca": [20],
        "DoorwayOrca": [10],
        "IntersectionOrca": [18],
    }
    following_distances = {
        "ClutterOrcaObs": [1.0, 1.5, 2.0, 2.5],
        "CorridorOrca": [1.0, 1.5, 2.0, 2.5],
        "DoorwayOrca": [1.0, 1.5, 2.0, 2.5],
        "IntersectionOrca": [1.0, 1.5, 2.0, 2.5],
    }

    plotted_following_distances = [1.0, 1.5, 2.0, 2.5]
    # plotted_human_nums = [1, 2, 3, 4]

    scenarios = [
        # "ClutterOrcaObs30H10",
        # "ClutterOrcaObs30H20",
        "ClutterOrcaObs30H30",
        # "ClutterOrcaObs30H40",

        # "CorridorOrcaH4W5.6",
        # "CorridorOrcaH12W5.6",
        "CorridorOrcaH20W5.6",
        # "CorridorOrcaH28W5.6",

        # "DoorwayOrcaH2W2.8",
        # "DoorwayOrcaH6W2.8",
        "DoorwayOrcaH10W2.8",
        # "DoorwayOrcaH14W2.8",

        # "IntersectionOrcaH2W4.8",
        # "IntersectionOrcaH10W4.8",
        "IntersectionOrcaH18W4.8",
        # "IntersectionOrcaH26W4.8",
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
    print("=== Evaluating Topography Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expLayoutFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "ClutterOrcaObs30H10",
        # "ClutterOrcaObs30H20",
        "ClutterOrcaObs30H30",
        # "ClutterOrcaObs30H40",
        ]
    # plotted_human_nums = [10, 20, 30, 40]
    print("=== Evaluating Clutter Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expLayoutClutterFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "CorridorOrcaH4W5.6",
        # "CorridorOrcaH12W5.6",
        "CorridorOrcaH20W5.6",
        # "CorridorOrcaH28W5.6",
        ]
    # plotted_human_nums = [4, 12, 20, 28]
    print("=== Evaluating Corridor Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expLayoutCorridorFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "DoorwayOrcaH2W2.8",
        # "DoorwayOrcaH6W2.8",
        "DoorwayOrcaH10W2.8",
        # "DoorwayOrcaH14W2.8",
        ]
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expLayoutDoorwayFDist", EVAL_RESULTS_DIR)

    scenarios = [
        # "IntersectionOrcaH2W4.8",
        # "IntersectionOrcaH10W4.8",
        "IntersectionOrcaH18W4.8",
        # "IntersectionOrcaH26W4.8",
        ]
    print("=== Evaluating Intersection Dist ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, following_distances, plotted_following_distances, following_positions, "expLayoutIntersectionFDist", EVAL_RESULTS_DIR)