#!/usr/bin/env python

import os
import sys

if "--headless" in sys.argv:
    os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import time

import irsim
import numpy as np
from tqdm import tqdm

# Repo root contains the Adap_RPF and traj_predictor packages (example/robot_person_following -> ../..).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Adap_RPF.adaptive_goal_generation import (
    AdaptiveGoalGenerator,
    MPCStats,
    traj_predictor_configs,
)
from Adap_RPF.mppi_local_control import MPPILocalController, mppi_default_config


def run_experiment(world_name, args, mppi_params=None):
    if args.log_path != "":
        eval_dir = os.path.join(args.log_path, os.path.basename(world_name).replace(".yaml", ""))
    else:
        eval_dir = ""

    enable_display = not args.headless
    enable_step_visuals = not args.headless

    env = irsim.make(
        world_name=world_name,
        save_ani=False,
        display=enable_display,
        full=False,
        eval_dir=eval_dir,
    )

    mpc_stats = MPCStats()
    base_config = traj_predictor_configs[args.traj_predictor].copy()
    base_config["dt"] = env.step_time
    if mppi_params:
        base_config["mppi"] = mppi_params

    goal_generator = AdaptiveGoalGenerator(predictor_type=args.traj_predictor, config=base_config)
    mppi_controller = MPPILocalController(
        config=base_config.get("mppi", mppi_default_config),
        dt=base_config["dt"],
        prediction_horizon=base_config.get("prediction_horizon", 2.0),
    )

    target_lost_count = 0
    result = "success"
    target_stopped_threshold = 0.05
    arrival_distance_threshold = 1.50

    if len(env.human_list) == 0:
        max_steps = args.min_steps
        max_search_steps = int(args.min_steps * 0.5)
    else:
        max_steps = args.min_steps * 2
        max_search_steps = int(args.min_steps * 0.5)

    for i in tqdm(range(max_steps)):
        robot_pose = env.robot.state
        target_pose = env.target.state
        target_velocity = env.target.velocity_xy

        robot_position = robot_pose[:2, 0]
        target_position = target_pose[:2, 0]

        target_visible = bool(getattr(env, "check_target_visible", True))
        if not target_visible:
            target_lost_count += 1

        target_vel_magnitude = np.linalg.norm(target_velocity)
        robot_target_distance = np.linalg.norm(target_position - robot_position)

        if target_vel_magnitude < target_stopped_threshold and robot_target_distance < arrival_distance_threshold:
            start_time = time.time()
            env.step(np.zeros((2, 1)))
            alg_cost_t = time.time() - start_time
            if args.visualize and not args.headless:
                env.save_figure(save_name="step_{:04d}.png".format(i))
            if enable_step_visuals:
                env.render(show_traj=True, show_trail=True)
            if eval_dir != "":
                env.record(alg_cost_t)
            if env.done():
                break
            continue

        start_time = time.time()
        try:
            goal_params = {"position": args.position, "distance": args.distance}
            goal_result = goal_generator.build_goal_trajectory(env, robot_pose, goal_params)
            goal_traj = goal_result["goal_traj"]
            predicted_traj = goal_result["predicted_traj"]

            if enable_step_visuals:
                if goal_result.get("mode") == "fallback_history":
                    if goal_result.get("visual_goal_traj") is not None:
                        env.draw_trajectory(goal_result["visual_goal_traj"], traj_type="k-", refresh=True)
                elif goal_result.get("mode") == "adaptive":
                    goal_generator.visualize_sample_points(env)
                    current_goal_pose = goal_result.get("current_goal_pose")
                    if current_goal_pose is not None:
                        env.draw_points(current_goal_pose.reshape(3, 1), s=50, c="red", refresh=True)
                    if goal_result.get("visual_goal_traj") is not None:
                        env.draw_trajectory(goal_result["visual_goal_traj"], traj_type="g--", refresh=True)

            action, mppi_info = mppi_controller.control(robot_pose, goal_traj, predicted_traj)
            if enable_step_visuals:
                try:
                    mppi_controller.visualize_mppi_samples(env)
                except Exception as exc:
                    print(f"WARN: MPPI sample trajectory visualization failed, skipping: {exc}")

            alg_cost_t = time.time() - start_time
            if mppi_info.get("success", False):
                mpc_stats.update_time(alg_cost_t, i + 1)
                env.step(np.array([[action[0]], [action[1]]]))
            else:
                mpc_stats.record_failure(i + 1, mppi_info.get("error", "Solver failed inside controller.control"))
                env.step(np.zeros((2, 1)))

        except Exception as exc:
            alg_cost_t = time.time() - start_time
            print(f"Control loop error: {exc}")
            import traceback

            traceback.print_exc()
            mpc_stats.record_failure(i + 1, str(exc))
            env.step(np.zeros((2, 1)))

        goal_generator.step_counter += 1

        if args.visualize and not args.headless:
            env.save_figure(save_name="step_{:04d}.png".format(i))

        if enable_step_visuals:
            env.render(show_traj=True, show_trail=True)

        if eval_dir != "":
            env.record(alg_cost_t)

        if env.robot.collision_flag:
            result = "collision"
            break

        if env.done():
            break

    if eval_dir != "":
        result_data = {
            "result": result,
            "steps_completed": i + 1,
            "time": env.step_time * (i + 1),
            "target_lost_count": target_lost_count,
            "mpc_stats": mpc_stats.get_stats_dict(),
            "takeover_pose_stats": goal_generator.takeover_pose_stats.get_stats_dict(),
        }
        with open(os.path.join(eval_dir, "experiment_result.json"), "w") as f:
            json.dump(result_data, f, indent=4)

        print("evaluating...")
        env.eval(max_steps=max_steps, max_search_steps=max_search_steps)

    if args.save_animation and not args.headless:
        env.end(ani_name="adap_rpf_mppi_planner_following", show_traj=True, show_trail=True, ending_time=1)

    return result, mpc_stats.get_stats_dict()


def main(world_name, args):
    if args.traj_predictor not in traj_predictor_configs:
        raise ValueError(f"Unsupported trajectory predictor: {args.traj_predictor}")

    mppi_params = {
        "safe_dist": args.safe_dist,
        "obstacle_cost_weight": args.obstacle_cost,
        "goal_cost_weight": args.goal_cost,
        "control_cost_weight": args.control_cost,
        "backward_cost_weight": args.backward_cost,
        "predictive_risk_weight": args.predictive_risk_weight,
        "noise": args.mppi_noise,
        "samples": args.mppi_samples,
    }
    return run_experiment(world_name, args, mppi_params=mppi_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adap-RPF goal generation with MPPI for differential-drive following")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="directory containing scenario yaml files")
    parser.add_argument("-s", "--scenario", type=str, default="square", help="scenario name")
    parser.add_argument("-p", "--position", type=str, default="back", help="back, left_side, right_side")
    parser.add_argument("-d", "--distance", type=float, default=1.5, help="distance to the target")
    parser.add_argument(
        "-m",
        "--min_steps",
        type=int,
        default=1500,
        help="minimum steps required for target completing the whole trajectory without any human",
    )
    parser.add_argument("-i", "--index", type=int, default=0, help="in this scenario, following position and distance, which index to run")
    parser.add_argument("-l", "--log_path", type=str, default="", help="evaluation log path")
    parser.add_argument("-t", "--traj_predictor", type=str, choices=["cv", "cvkf"], default="cv", help="trajectory predictor (cv, cvkf)")
    parser.add_argument("-v", "--visualize", action="store_true", default=False, help="whether to save per-step visualization frames")
    parser.add_argument("--headless", action="store_true", default=False, help="disable GUI display and per-step visualization for faster evaluation")
    parser.add_argument("--save_animation", action="store_true", default=False, help="save an animation when GUI mode is enabled")

    parser.add_argument("--safe_dist", type=float, default=1.2, help="MPPI safety distance")
    parser.add_argument("--obstacle_cost", type=float, default=4.0, help="MPPI obstacle cost weight")
    parser.add_argument("--goal_cost", type=float, default=1.0, help="MPPI goal cost weight")
    parser.add_argument("--control_cost", type=float, default=0.1, help="MPPI control cost weight")
    parser.add_argument("--backward_cost", type=float, default=0.0, help="MPPI backward cost weight")
    parser.add_argument("--predictive_risk_weight", type=float, default=0.0, help="MPPI predictive risk weight")
    parser.add_argument("--mppi_noise", nargs=2, type=float, default=[2.0, 0.5], help="MPPI noise parameters")
    parser.add_argument("--mppi_samples", type=int, default=500, help="MPPI sample count")

    args = parser.parse_args()

    indexed_env_path_file = os.path.join(
        args.config_path,
        args.scenario + "_" + args.position + "_" + str(args.index) + ".yaml",
    )
    base_env_path_file = os.path.join(args.config_path, args.scenario + "_" + args.position + ".yaml")
    env_path_file = indexed_env_path_file if os.path.exists(indexed_env_path_file) else base_env_path_file

    main(env_path_file, args)
