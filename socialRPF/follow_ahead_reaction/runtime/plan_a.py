import os
import shutil
import sys
from pathlib import Path

if "--headless" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)

import numpy as np

from .base import irsim, pm, resolve_config_dir
from .framework import (
    FollowAheadReactionIRSimFramework,
    build_parser,
)
from .fair_adapter import compute_baseline_eval_horizons


class FollowAheadReactionIRSimPlanA(FollowAheadReactionIRSimFramework):
    """Baseline-aligned Follow-Reaction with original fixed action semantics.

    Plan A keeps the fair-framework evaluation/runtime alignment, but removes the
    current benchmark-specific runtime motion supervisor:
    - no back/side dynamic speed retuning
    - no action downgrading before execution
    - restore the original fixed robot/human motion parameters used by the
      Follow_ahead_reaction ROS implementation
    """

    def __init__(self, env, args):
        super().__init__(env, args)
        self.original_robot_vel = 0.6
        self.original_fast_lambda = 1.5
        self.original_human_vel = 0.6
        self.original_robot_angle = 45.0
        self.original_human_angle = 10.0
        self._restore_original_motion_params()

    def _restore_original_motion_params(self):
        self.params["robot_vel"] = float(self.original_robot_vel)
        self.params["robot_vel_fast_lamda"] = float(self.original_fast_lambda)
        self.params["human_vel"] = float(self.original_human_vel)
        self.params["robot_angle"] = float(self.original_robot_angle)
        self.params["human_angle"] = float(self.original_human_angle)

    def update_back_motion_params(self, state):
        self._restore_original_motion_params()

    def update_side_motion_params(self, state):
        self._restore_original_motion_params()

    def command_from_action(self):
        action = self.best_action

        linear_vel = self.params["robot_vel"]
        if action in {"fast_straight", "fast_right", "fast_left"}:
            linear_vel *= self.params["robot_vel_fast_lamda"]

        angular_vel = self.params["robot_angle"] * np.pi / 180.0
        if action in {"straight", "fast_straight"}:
            angular_vel = 0.0
        elif action in {"right", "fast_right"}:
            angular_vel *= -1.0
        elif action in {"stop", None}:
            linear_vel = 0.0
            angular_vel = 0.0

        return np.array([[linear_vel], [angular_vel]])


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
    controller = FollowAheadReactionIRSimPlanA(env, args)

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


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    config_dir = resolve_config_dir(args.config_path)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    scenario_name = f"{args.scenario}_{args.position}_{args.index}"
    world_name = config_dir / f"{scenario_name}.yaml"
    if not world_name.exists():
        raise FileNotFoundError(f"Scenario yaml not found: {world_name}")

    main(str(world_name), args)
