import matplotlib.pyplot as plt
import numpy as np
import os
import json
from constants import METHODS, COLORS, EVALUATED_METRICS, FOLLOWING_POSITIONS, LINE_STYLES, LOGS_STATIC_1_5M, EVAL_RESULTS_DIR

def load_data(base_dir, method, widths, plotted_widths, following_positions, scenarios, trials, evaluated_metrics):
    # initialize a dictionary to hold evaluation data
    evaluation_data = {}
    for following_position in following_positions:
        evaluation_data[following_position] = {}
        for plotted_width in plotted_widths:
            evaluation_data[following_position][plotted_width] = {}
            for metric in evaluated_metrics:
                evaluation_data[following_position][plotted_width][metric] = []

    for following_position in following_positions:
        for i, plotted_width in enumerate(plotted_widths):
            for scenario in scenarios:
                for key, value in widths.items():
                    if key in scenario:
                        real_widths = value
                if "{}".format(real_widths[i]) in scenario:
                    for trial in trials:
                        file_path = os.path.join(
                            base_dir,
                            method,
                            f"{scenario}_{following_position}_{trial}",
                            "eval_result.json"
                        )
                        if not os.path.exists(file_path):
                            print(f"File not found: {file_path}")
                            continue
                        else:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                for metric in evaluated_metrics:
                                    if metric in data:
                                        evaluation_data[following_position][plotted_width][metric].append(data[metric])
    return evaluation_data

def plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, widths, plotted_widths, following_positions, saved_fname, output_dir):
    evaluated_data = {}
    # method -> following_position -> human_num -> metric -> values
    for method in methods.keys():
        evaluated_data[method] = load_data(base_dir, method, widths, plotted_widths, following_positions, scenarios, trials, evaluated_metrics)

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
                for plotted_width in plotted_widths:
                    if plotted_metric in ["ASR", "TVR"]:
                        data = np.array(evaluated_data[method][following_position][plotted_width][value[0]]) * 100
                    elif plotted_metric == "SR":
                        data = np.array(evaluated_data[method][following_position][plotted_width][value[0]]) * np.array(evaluated_data[method][following_position][plotted_width][value[1]]) * 100
                    elif plotted_metric in ["PL", "AvgVel", "AvgAcc", "Jerk"]:
                        data = np.array(evaluated_data[method][following_position][plotted_width][value[0]])
                    elif plotted_metric == "TinTPerson" or plotted_metric == "TinPrivate":
                        data = np.array(evaluated_data[method][following_position][plotted_width][value[0]]) / (np.array(evaluated_data[method][following_position][plotted_width][value[1]])*0.1) * 100
                    else:
                        raise ValueError(f"Unsupported plotted metric: {plotted_metric}")
                    if len(data) > 0:
                        plotted_x.append(plotted_width)
                        plotted_values.append(np.mean(data))

                if len(plotted_values) > 0:
                    if plotted_metric in ["ASR", "TVR", "SR", "TinTPerson", "TinPrivate"]:
                        ax.plot(plotted_x, plotted_values, marker='o', color=colors[method], linestyle=line_styles[following_position], label=f"{label}-{following_position}")
                        ax.set_ylim(0, 105)
                    else:
                        ax.plot(plotted_x, plotted_values, marker='o', color=colors[method], linestyle=line_styles[following_position], label=f"{label}-{following_position}")
                        ax.set_ylim(0, 45)
                
                # print(f"[{method}-{following_position}] {plotted_metric}:")
                # print(f"  Widths: {plotted_widths}, Values: {plotted_values}")

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





if __name__ == "__main__":
    """
    1. Visualize each following direction separately.
    2. Compare all methods in the same figure.
    3. Average across all scenarios.
    """
    base_dir = LOGS_STATIC_1_5M
    following_positions = FOLLOWING_POSITIONS
    methods = METHODS
    colors = COLORS
    line_styles = LINE_STYLES
    evaluated_metrics = EVALUATED_METRICS

    ### Topographies ###
    widths = {
        'ClutterOrcaObs': [10, 20, 30, 40],
        'CorridorOrca': [6.8, 6.2, 5.6, 5.0],
        'DoorwayOrca': [3.4, 3.1, 2.8, 2.5],
        'IntersectionOrca': [6.0, 5.4, 4.8, 4.2]
    }
    plotted_widths = [1, 2, 3, 4]

    scenarios = [
        "ClutterOrcaObs10H30",
        "ClutterOrcaObs20H30",
        "ClutterOrcaObs30H30",
        "ClutterOrcaObs40H30",

        "CorridorOrcaH20W5.0",
        "CorridorOrcaH20W5.6",
        "CorridorOrcaH20W6.2",
        "CorridorOrcaH20W6.8",

        "DoorwayOrcaH10W2.5",
        "DoorwayOrcaH10W2.8",
        "DoorwayOrcaH10W3.1",
        "DoorwayOrcaH10W3.4",

        "IntersectionOrcaH18W4.2",
        "IntersectionOrcaH18W4.8",
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
    print("=== Evaluating Topography Width ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, widths, plotted_widths, following_positions, "expLayoutOccupancy", EVAL_RESULTS_DIR)

    scenarios = [
        "ClutterOrcaObs10H30",
        "ClutterOrcaObs20H30",
        "ClutterOrcaObs30H30",
        "ClutterOrcaObs40H30",
        ]
    widths = {
        'ClutterOrcaObs': [10, 20, 30, 40],
    }
    trials = [i for i in range(1, 101)]
    plotted_widths = [1, 2, 3, 4]
    print("=== Evaluating Clutter Obstacle ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, widths, plotted_widths, following_positions, "expLayoutClutterOccupancy", EVAL_RESULTS_DIR)

    scenarios = [
        "CorridorOrcaH20W5.0",
        "CorridorOrcaH20W5.6",
        "CorridorOrcaH20W6.2",
        "CorridorOrcaH20W6.8",
        ]
    widths = {
        'CorridorOrca': [6.8, 6.2, 5.6, 5.0],
    }
    trials = [i for i in range(1, 101)]
    plotted_widths = [1, 2, 3, 4]
    print("=== Evaluating Corridor Widths ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, widths, plotted_widths, following_positions, "expLayoutCorridorOccupancy", EVAL_RESULTS_DIR)

    scenarios = [
        "DoorwayOrcaH10W2.5",
        "DoorwayOrcaH10W2.8",
        "DoorwayOrcaH10W3.1",
        "DoorwayOrcaH10W3.4",
        ]
    widths = {
        'DoorwayOrca': [3.4, 3.1, 2.8, 2.5],
    }
    trials = [i for i in range(1, 101)]
    plotted_widths = [1, 2, 3, 4]
    print("=== Evaluating Doorway Widths ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, widths, plotted_widths, following_positions, "expLayoutDoorwayOccupancy", EVAL_RESULTS_DIR)


    scenarios = [
        "IntersectionOrcaH18W4.2",
        "IntersectionOrcaH18W4.8",
        "IntersectionOrcaH18W5.4",
        "IntersectionOrcaH18W6.0",
        ]
    widths = {
        'IntersectionOrca': [6.0, 5.4, 4.8, 4.2],
    }
    trials = [i for i in range(1, 101)]
    plotted_widths = [1, 2, 3, 4]
    print("=== Evaluating Intersection Widths ===")
    plot_results(base_dir, methods, scenarios, trials, evaluated_metrics, plotted_metrics, colors, line_styles, widths, plotted_widths, following_positions, "expLayoutIntersectionOccupancy", EVAL_RESULTS_DIR)