import matplotlib.pyplot as plt
import numpy as np
import os
import json
from constants import METHODS, COLORS, EVALUATED_METRICS, FOLLOWING_POSITIONS, LINE_STYLES, LOGS_SFM, EVAL_RESULTS_DIR

def load_data(base_dir, method, human_nums, plotted_human_nums, following_positions, scenarios, trials, evaluated_metrics):
    # initialize a dictionary to hold evaluation data
    evaluation_data = {}
    for following_position in following_positions:
        evaluation_data[following_position] = {}
        for plotted_human_num in plotted_human_nums:
            evaluation_data[following_position][plotted_human_num] = {}
            for metric in evaluated_metrics:
                evaluation_data[following_position][plotted_human_num][metric] = []

    for following_position in following_positions:
        for i, plotted_human_num in enumerate(plotted_human_nums):
            for scenario in scenarios:
                for key, value in human_nums.items():
                    if key in scenario:
                        real_human_nums = value
                if "H{}".format(real_human_nums[i]) in scenario:
                    for trial in trials:
                        file_path = os.path.join(
                            base_dir,
                            method,
                            f"{scenario}_{following_position}_{trial}",
                            "eval_result.json"
                        )
                        if not os.path.exists(file_path):
                            # print(f"File not found: {file_path}")
                            continue
                        else:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                for metric in evaluated_metrics:
                                    if metric in data:
                                        evaluation_data[following_position][plotted_human_num][metric].append(data[metric])
                                        if metric == "target_visibility_ratio" and data[metric] > 1.0:
                                            print(f"Warning: target_visibility_ratio > 1.0 in {file_path}")
    return evaluation_data

def plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, saved_fname, output_dir):
    evaluated_data = {}
    # method -> following_position -> human_num -> metric -> values
    for method in methods.keys():
        evaluated_data[method] = load_data(base_dir, method, human_nums, plotted_human_nums, following_positions, scenarios, trials, evaluated_metrics)

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
                for human_num in plotted_human_nums:
                    if plotted_metric in ["ASR", "TVR"]:
                        data = np.array(evaluated_data[method][following_position][human_num][value[0]]) * 100
                    elif plotted_metric == "SR":
                        data = np.array(evaluated_data[method][following_position][human_num][value[0]]) * np.array(evaluated_data[method][following_position][human_num][value[1]]) * 100
                    elif plotted_metric in ["PL", "AvgVel", "AvgAcc", "Jerk"]:
                        data = np.array(evaluated_data[method][following_position][human_num][value[0]])
                    elif plotted_metric == "TinTPerson" or plotted_metric == "TinPrivate":
                        data = np.array(evaluated_data[method][following_position][human_num][value[0]]) / (np.array(evaluated_data[method][following_position][human_num][value[1]])*0.1) * 100
                    else:
                        raise ValueError(f"Unsupported plotted metric: {plotted_metric}")
                    if len(data) > 0:
                        plotted_x.append(human_num)
                        plotted_values.append(np.mean(data))
                        if method == "rda_planner_diff" or method == "rda_traj_planner_diff":
                            print(f"Method: {method}, Position: {following_position}, Humans: {human_num}, Metric: {plotted_metric}, Value: {np.mean(data):.2f}")

                if len(plotted_values) > 0:
                    if plotted_metric in ["ASR", "TVR", "SR", "TinTPerson", "TinPrivate"]:
                        ax.plot(plotted_x, plotted_values, marker='o', color=colors[method], linestyle=line_styles[following_position], label=f"{label}-{following_position}")
                        ax.set_ylim(0, 105)
                    else:
                        ax.plot(plotted_x, plotted_values, marker='o', color=colors[method], linestyle=line_styles[following_position], label=f"{label}-{following_position}")
                        ax.set_ylim(0, 45)

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
    plt.savefig(os.path.join(output_dir, "{}.png".format(saved_fname)), dpi=500)
    plt.show()





if __name__ == "__main__":
    """
    1. Visualize each following direction separately.
    2. Compare all methods in the same figure.
    3. Average across all scenarios.
    """
    base_dir = LOGS_SFM
    following_positions = FOLLOWING_POSITIONS
    methods = METHODS
    colors = COLORS
    line_styles = LINE_STYLES
    evaluated_metrics = EVALUATED_METRICS


    ### Dynamic Crowds (Sfm) ###
    scenarios = [
        "NormalCrowdSfmH5W0",
        "NormalCrowdSfmH10W0",
        "NormalCrowdSfmH15W0",
        "NormalCrowdSfmH20W0",
        "NormalCrowdSfmH25W0",
        "NormalCrowdSfmH30W0",

        "NormalCircularSfmH5W0",
        "NormalCircularSfmH10W0",
        "NormalCircularSfmH15W0",
        "NormalCircularSfmH20W0",
        "NormalCircularSfmH25W0",
        "NormalCircularSfmH30W0",

        "NormalParallelSfmH5W0",
        "NormalParallelSfmH10W0",
        "NormalParallelSfmH15W0",
        "NormalParallelSfmH20W0",
        "NormalParallelSfmH25W0",
        "NormalParallelSfmH30W0",

        "NormalPerpendicularSfmH5W0",
        "NormalPerpendicularSfmH10W0",
        "NormalPerpendicularSfmH15W0",
        "NormalPerpendicularSfmH20W0",
        "NormalPerpendicularSfmH25W0",
        "NormalPerpendicularSfmH30W0",
        ]
    
    human_nums = {
        "NormalCrowdSfm": [5, 10, 15, 20, 25, 30],
        "NormalCircularSfm": [5, 10, 15, 20, 25, 30],
        "NormalParallelSfm": [5, 10, 15, 20, 25, 30],
        "NormalPerpendicularSfm": [5, 10, 15, 20, 25, 30],
    }

    plotted_human_nums = [5, 10, 15, 20, 25, 30]

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
    print("=== Evaluating Dynamic Crowd Human ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expDynaCrowdHumanSFM", EVAL_RESULTS_DIR)

    scenarios = [
        "NormalCrowdSfmH5W0",
        "NormalCrowdSfmH10W0",
        "NormalCrowdSfmH15W0",
        "NormalCrowdSfmH20W0",
        "NormalCrowdSfmH25W0",
        "NormalCrowdSfmH30W0",
        ]
    print("=== Evaluating Crowd Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expCrowdHumanSFM")

    scenarios = [
        "NormalCircularSfmH5W0",
        "NormalCircularSfmH10W0",
        "NormalCircularSfmH15W0",
        "NormalCircularSfmH20W0",
        "NormalCircularSfmH25W0",
        "NormalCircularSfmH30W0",
        ]
    print("=== Evaluating Circular Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expCircularHumanSFM")

    scenarios = [
        "NormalParallelSfmH5W0",
        "NormalParallelSfmH10W0",
        "NormalParallelSfmH15W0",
        "NormalParallelSfmH20W0",
        "NormalParallelSfmH25W0",
        "NormalParallelSfmH30W0",
        ]
    print("=== Evaluating Parallel Human ===")
    # trials = [i for i in range(1, 21)]
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expParallelHumanSFM")

    scenarios = [
        "NormalPerpendicularSfmH5W0",
        "NormalPerpendicularSfmH10W0",
        "NormalPerpendicularSfmH15W0",
        "NormalPerpendicularSfmH20W0",
        "NormalPerpendicularSfmH25W0",
        "NormalPerpendicularSfmH30W0",
        ]
    print("=== Evaluating Perpendicular Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expPerpendicularHumanSFM")

    ### Topographies ###
    human_nums = {
        "ClutterSfmObs": [10, 20, 30, 40],
        "CorridorSfm": [4, 12, 20, 28],
        "DoorwaySfm": [2, 6, 10, 14],
        "IntersectionSfm": [2, 10, 18, 26],
    }
    plotted_human_nums = [1, 2, 3, 4]

    # scenarios = [
    #     "ClutterSfmObs30H10",
    #     "ClutterSfmObs30H20",
    #     "ClutterSfmObs30H30",
    #     "ClutterSfmObs30H40",

    #     "CorridorSfmH4W5.6",
    #     "CorridorSfmH12W5.6",
    #     "CorridorSfmH20W5.6",
    #     "CorridorSfmH28W5.6",

    #     "DoorwaySfmH2W2.8",
    #     "DoorwaySfmH6W2.8",
    #     "DoorwaySfmH10W2.8",
    #     "DoorwaySfmH14W2.8",

    #     "IntersectionSfmH2W4.8",
    #     "IntersectionSfmH10W4.8",
    #     "IntersectionSfmH18W4.8",
    #     "IntersectionSfmH26W4.8",
    #     ]
    # trials = [i for i in range(1, 101)]
    # plotted_metrics = {
    #     "SR": ["obstacle_avoidance_success", "search_success", "%"],
    #     "ASR": ["obstacle_avoidance_success", "%"],
    #     "TVR": ["target_visibility_ratio", "%"],
    #     "TinTPerson": ["time_in_target_personal_zone", "max_steps", "%"],
    #     "TinPrivate": ["time_in_human_private_zone", "max_steps", "%"],
    #     # "SPLs": ["search_path_length", ""],
    #     # "PL": ["path_length", "m"],
    #     # "AvgVel": ["avg_velocity", "m/s"],
    #     # "AvgAcc": ["avg_acceleration", "m/s$^2$"],
    #     "Jerk": ["avg_jerk", "m/s$^3$"],
    # }
    # print("=== Evaluating Topography Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expTopographyHuman")

    # scenarios = [
    #     "ClutterSfmObs30H10",
    #     "ClutterSfmObs30H20",
    #     "ClutterSfmObs30H30",
    #     "ClutterSfmObs30H40",
    #     ]
    # trials = [i for i in range(1, 101)]
    # plotted_human_nums = [10, 20, 30, 40]
    # print("=== Evaluating Clutter Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expClutterHuman")

    # scenarios = [
    #     "CorridorSfmH4W5.6",
    #     "CorridorSfmH12W5.6",
    #     "CorridorSfmH20W5.6",
    #     "CorridorSfmH28W5.6",
    #     ]
    # trials = [i for i in range(1, 101)]
    # plotted_human_nums = [4, 12, 20, 28]
    # print("=== Evaluating Corridor Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expCorridorHuman")

    # scenarios = [
    #     "DoorwaySfmH2W2.8",
    #     "DoorwaySfmH6W2.8",
    #     "DoorwaySfmH10W2.8",
    #     "DoorwaySfmH14W2.8",
    #     ]
    # trials = [i for i in range(1, 101)]
    # plotted_human_nums = [2, 6, 10, 14]
    # print("=== Evaluating Doorway Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expDoorwayHuman")

    # scenarios = [
    #     "IntersectionSfmH2W4.8",
    #     "IntersectionSfmH10W4.8",
    #     "IntersectionSfmH18W4.8",
    #     "IntersectionSfmH26W4.8",
    #     ]
    # trials = [i for i in range(1, 101)]
    # plotted_human_nums = [2, 10, 18, 26]
    # print("=== Evaluating Intersection Human ===")
    # plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, human_nums, plotted_human_nums, following_positions, "expIntersectionHuman")