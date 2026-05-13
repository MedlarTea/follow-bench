import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

if "--headless" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)

from .base import irsim, pm, resolve_config_dir
from .strict_visible import (
    FollowAheadReactionIRSimStrictVisible,
)
from .fair_adapter import (
    build_perception_costmap_from_env,
    compute_baseline_eval_horizons,
)


class FollowAheadReactionIRSimFramework(FollowAheadReactionIRSimStrictVisible):
    """Follow-Reaction under a baseline-aligned evaluation framework.

    What stays from Follow-Reaction:
    - LSTM human action predictor
    - RL value model
    - MCTS action search
    - reward / action set / motion policy

    What is aligned to the other baseline scripts:
    - hidden-target failure semantics via max_search_steps in env.eval
    - robot-centric lidar + visible-human obstacle input
    - no global costmap oracle for MCTS obstacle checking
    """

    def __init__(self, env, args):
        super().__init__(env, args)
        self.perception_costmap = None
        self.perception_resolution = 0.1
        self.perception_inflation_radius = 0.55
        self.refresh_perception_costmap(self.get_robot_state())

    def refresh_perception_costmap(self, robot_state):
        map_params, costmap = build_perception_costmap_from_env(
            self.env,
            robot_state,
            resolution=self.perception_resolution,
            inflation_radius=self.perception_inflation_radius,
        )
        self.params.update(map_params)
        self.perception_costmap = costmap
        self.map_height = costmap.shape[0]

        if self.args.show_debug:
            print(
                f"[PERCEPTION-MAP] step={self.step_count} "
                f"width={self.params['map_width']} "
                f"height={self.map_height} "
                f"res={self.params['map_res']:.3f} "
                f"scan_points={self.params['perception_scan_points']} "
                f"visible_humans={self.params['perception_visible_humans']}"
            )

    def step(self):
        self.step_count += 1
        state = self.current_state()
        self.update_back_motion_params(state)
        self.update_side_motion_params(state)
        self.refresh_perception_costmap(state[0, :])

        executed_action = self.best_action
        control = self.command_from_action()
        alg_cost_t = self.maybe_plan(state)
        self.draw_debug(state)

        if self.args.show_debug:
            print(
                f"[CMD] step={self.step_count} "
                f"v={float(control[0, 0]):.3f} "
                f"w={float(control[1, 0]):.3f} "
                f"executed_action={executed_action} "
                f"planned_action={self.best_action}"
            )
        return control, alg_cost_t


def main(world_name, args):
    world_path = Path(world_name)
    if not world_path.exists():
        raise FileNotFoundError(f"Scenario yaml not found: {world_path}")

    if args.save_animation and os.path.isdir(pm.ani_buffer_path):
        shutil.rmtree(pm.ani_buffer_path)

    if args.debug_reward:
        os.environ["FOLLOW_AHEAD_DEBUG_REWARD"] = "1"
    else:
        os.environ.pop("FOLLOW_AHEAD_DEBUG_REWARD", None)

    if args.debug_obstacle:
        os.environ["FOLLOW_AHEAD_DEBUG_OBSTACLE"] = "1"
    else:
        os.environ.pop("FOLLOW_AHEAD_DEBUG_OBSTACLE", None)

    if args.log_path:
        eval_dir = os.path.join(args.log_path, world_path.stem)
    else:
        eval_dir = ""

    env = irsim.make(
        world_name=str(world_path),
        save_ani=args.save_animation,
        display=not args.headless,
        full=False,
        eval_dir=eval_dir,
    )
    controller = FollowAheadReactionIRSimFramework(env, args)

    max_steps, max_search_steps = compute_baseline_eval_horizons(env, args.min_steps)
    if args.show_debug:
        print(
            f"[EVAL] min_steps={args.min_steps} "
            f"max_steps={max_steps} "
            f"max_search_steps={max_search_steps}"
        )

    for step in range(max_steps):
        control, alg_cost_t = controller.step()

        if args.visualize:
            env.save_figure(save_name=f"step_{step:04d}.png")

        env.step(control)
        env.render(show_traj=True, show_trail=True)

        if eval_dir:
            env.record(alg_cost_t)

        if env.done():
            break

    if eval_dir:
        print("evaluating...")
        env.eval(max_steps=max_steps, max_search_steps=max_search_steps)

    env.end(
        ending_time=0.0 if args.headless else 3.0,
        ani_name=world_path.stem,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="directory containing scenario yaml files",
    )
    parser.add_argument("-s", "--scenario", type=str, default="triangle")
    parser.add_argument(
        "-p",
        "--position",
        type=str,
        default="front",
        help="which yaml suffix to use: front/back/left_side/right_side",
    )
    parser.add_argument(
        "-m",
        "--min_steps",
        type=int,
        default=500,
        help="baseline-aligned minimum episode steps; eval horizons are derived from this",
    )
    parser.add_argument(
        "-d",
        "--follow_distance",
        type=float,
        default=1.5,
        help="desired follow distance used for reward shaping and RL model selection",
    )
    parser.add_argument(
        "--rl_obs_mode",
        type=str,
        default="relative_pose",
        choices=["relative_pose", "task_error"],
        help="observation encoding used by the RL value model",
    )
    parser.add_argument(
        "--rl_model_override",
        type=str,
        default="",
        help="optional path to a specific RL value model zip used for this run only",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="scenario index used for evaluation yaml names",
    )
    parser.add_argument(
        "-l",
        "--log_path",
        type=str,
        default="",
        help="evaluation output directory",
    )
    parser.add_argument(
        "-t",
        "--traj_predictor",
        type=str,
        default="cvkf",
        help="reserved for CLI parity with other baselines; unused in this Follow-Reaction variant",
    )
    parser.add_argument(
        "--expansion_time",
        type=float,
        default=0.15,
        help="MCTS wall-time budget per replan",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="disable GUI rendering",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        default=False,
        help="save every frame to disk",
    )
    parser.add_argument(
        "--show_debug",
        action="store_true",
        default=False,
        help="print runtime diagnostics",
    )
    parser.add_argument(
        "--debug_reward",
        action="store_true",
        default=False,
        help="print reward decomposition from navState",
    )
    parser.add_argument(
        "--debug_obstacle",
        action="store_true",
        default=False,
        help="print occupancy hits from nodes.any_obs",
    )
    parser.add_argument(
        "--save_animation",
        "--save-animation",
        action="store_true",
        dest="save_animation",
        help="enable gif generation at the end of the run",
    )
    parser.add_argument(
        "--no_save_animation",
        "--no-save-animation",
        action="store_false",
        dest="save_animation",
        help="disable gif generation at the end of the run",
    )
    parser.set_defaults(save_animation=False)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    config_dir = resolve_config_dir(args.config_path)
    if args.log_path:
        env_path_file = config_dir / f"{args.scenario}_{args.position}_{args.index}.yaml"
    else:
        env_path_file = config_dir / f"{args.scenario}_{args.position}.yaml"

    main(str(env_path_file), args)
