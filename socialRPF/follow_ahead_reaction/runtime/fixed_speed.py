import os
import shutil
import sys
from pathlib import Path

if "--headless" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)

from .base import irsim, pm, resolve_config_dir
from .plan_a import (
    FollowAheadReactionIRSimPlanA,
)
from .framework import build_parser
from .fair_adapter import compute_baseline_eval_horizons


class FollowAheadReactionIRSimFixedSpeed(
    FollowAheadReactionIRSimPlanA
):
    """Plan A variant with fixed robot speeds 1.0 / 1.5.

    This keeps the fair framework and Plan A's fixed action semantics, but raises
    the robot action magnitudes from:
    - normal: 0.6 -> 1.0
    - fast  : 0.9 -> 1.5

    Human motion parameters remain unchanged so the comparison is still a
    Follow-Reaction speed-tuning variant under the same fair evaluation setup.
    """

    def __init__(self, env, args):
        super().__init__(env, args)
        self.original_robot_vel = 1.0
        self.original_fast_lambda = 1.5
        self._restore_original_motion_params()


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
    controller = FollowAheadReactionIRSimFixedSpeed(env, args)

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
