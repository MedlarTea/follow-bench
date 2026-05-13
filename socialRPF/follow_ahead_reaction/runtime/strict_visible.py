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

from .base import (
    FollowAheadReactionIRSim,
    irsim,
    pm,
    resolve_config_dir,
)


class FollowAheadReactionIRSimStrictVisible(FollowAheadReactionIRSim):
    """A stricter benchmark-aligned variant.

    Key difference from the default script:
    - Only uses target ground truth while the target is visible.
    - Once the target is not visible, it keeps an internal target estimate and
      rolls it forward using the original LSTM human action predictor.

    This keeps the overall Follow-Reaction planning stack unchanged:
    - same LSTM action predictor
    - same RL value model
    - same MCTS controller
    - same evaluate.sh interface
    """

    def __init__(self, env, args):
        super().__init__(env, args)
        self.estimated_human_state = None
        self.last_visible_human_state = None
        self.last_visible_target_speed = self.params["human_vel"]
        self.predicted_human_history = []
        self.visibility_was_lost = False

    def _target_visible(self):
        return bool(getattr(self.env, "check_target_visible", False))

    def _default_human_prob(self):
        return {"left": 0.0, "straight": 1.0, "right": 0.0}

    def _get_visible_human_state(self):
        visible_state = super().get_human_state()
        self.last_visible_human_state = np.array(visible_state, dtype=float).copy()
        self.estimated_human_state = np.array(visible_state, dtype=float).copy()
        self.last_visible_target_speed = max(float(self.last_target_speed), 1e-3)
        self.predicted_human_history = []
        self.visibility_was_lost = False
        return visible_state

    def _seed_estimated_human_state(self):
        if self.estimated_human_state is not None:
            return np.array(self.estimated_human_state, dtype=float).copy()

        if self.last_visible_human_state is not None:
            return np.array(self.last_visible_human_state, dtype=float).copy()

        if self.human_history:
            seeded = np.zeros(3, dtype=float)
            seeded[:2] = np.asarray(self.human_history[-1], dtype=float)
            seeded[2] = self.last_target_used_yaw
            return seeded

        # Strictly avoid peeking at env.target truth before visibility is gained.
        seeded = self.get_robot_state().copy()
        seeded[2] = self.last_target_used_yaw
        return seeded

    def _history_points_for_prediction(self):
        points = list(self.human_history)
        if self.predicted_human_history:
            points.extend(self.predicted_human_history)
        return points

    def _build_bootstrap_history_from_points(self, points):
        if len(points) < self.history_bootstrap_min_points:
            return None

        history = np.asarray(points, dtype=float)
        if len(history) >= 2:
            step_vec = np.mean(np.diff(history, axis=0), axis=0)
        else:
            yaw = self.last_target_used_yaw
            step_vec = (
                self.last_visible_target_speed
                * self.plan_interval
                * np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
            )

        pad_count = self.human_history_length - len(history)
        if pad_count <= 0:
            return history.tolist()[-self.human_history_length :]

        anchor = history[0]
        padding = [anchor - step_vec * (pad_count - idx) for idx in range(pad_count)]
        bootstrap_history = np.vstack([padding, history])
        return bootstrap_history.tolist()

    def _planning_history_for_estimation(self):
        points = self._history_points_for_prediction()
        if not points:
            return None

        if len(points) >= self.human_history_length:
            return points[-self.human_history_length :]

        return self._build_bootstrap_history_from_points(points)

    def _predict_human_prob_from_history(self):
        planning_history = self._planning_history_for_estimation()
        if planning_history is None:
            return self._default_human_prob()

        try:
            return self.human_prob.forward(planning_history)
        except Exception:
            return self._default_human_prob()

    def _propagate_estimated_human_state(self, human_state, human_prob):
        propagated = np.array(human_state, dtype=float).copy()
        dt = float(self.env.step_time)
        turn_angle = self.params["human_angle"] * np.pi / 180.0

        p_left = float(human_prob.get("left", 0.0))
        p_straight = float(human_prob.get("straight", 1.0))
        p_right = float(human_prob.get("right", 0.0))

        headings = np.array(
            [
                propagated[2] + turn_angle,
                propagated[2],
                propagated[2] - turn_angle,
            ],
            dtype=float,
        )
        unit_dirs = np.stack([np.cos(headings), np.sin(headings)], axis=1)
        blended_dir = (
            p_left * unit_dirs[0]
            + p_straight * unit_dirs[1]
            + p_right * unit_dirs[2]
        )

        norm = np.linalg.norm(blended_dir)
        if norm < 1e-8:
            blended_dir = unit_dirs[1]
            norm = 1.0
        blended_dir /= norm

        yaw = self.wrap_to_pi(float(np.arctan2(blended_dir[1], blended_dir[0])))
        speed = max(float(self.last_visible_target_speed), 1e-3)
        propagated[0] += speed * dt * blended_dir[0]
        propagated[1] += speed * dt * blended_dir[1]
        propagated[2] = self.quantize_human_yaw(yaw)

        self.last_target_speed = speed
        self.last_target_raw_yaw = yaw
        self.last_target_yaw = yaw
        self.last_target_used_yaw = float(propagated[2])
        return propagated

    def get_human_state(self):
        if self._target_visible():
            return self._get_visible_human_state()

        if not self.visibility_was_lost and self.args.show_debug:
            print(
                f"[TARGET] step={self.step_count} target lost; switching to history-only prediction"
            )
        self.visibility_was_lost = True

        human_prob = self._predict_human_prob_from_history()
        seeded_state = self._seed_estimated_human_state()
        self.estimated_human_state = self._propagate_estimated_human_state(
            seeded_state, human_prob
        )
        return np.array(self.estimated_human_state, dtype=float).copy()

    def get_human_history_point(self):
        if self.state is not None and self.state.shape == (2, 3):
            return self.state[1, :2].tolist()

        if self.estimated_human_state is not None:
            return self.estimated_human_state[:2].tolist()

        if self.last_visible_human_state is not None:
            return self.last_visible_human_state[:2].tolist()

        return None

    def update_human_history(self):
        if self._target_visible():
            point = self.get_human_history_point()
            if point is None:
                return len(self.human_history) > self.human_history_length

            self.human_history.append(point)
            self.predicted_human_history = []
            if len(self.human_history) <= self.human_history_length:
                return False

            self.human_history.pop(0)
            return True

        point = self.get_human_history_point()
        if point is not None:
            self.predicted_human_history.append(point)
            if len(self.predicted_human_history) > self.human_history_length:
                self.predicted_human_history.pop(0)

        return len(self.human_history) > self.history_bootstrap_min_points

    def build_bootstrap_history(self):
        return self._build_bootstrap_history_from_points(
            self._history_points_for_prediction()
        )

    def get_planning_history(self, history_ready):
        points = self._history_points_for_prediction()

        if len(points) >= self.human_history_length:
            return points[-self.human_history_length :]

        if len(points) >= self.history_bootstrap_min_points:
            return self._build_bootstrap_history_from_points(points)

        return None


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
    controller = FollowAheadReactionIRSimStrictVisible(env, args)

    max_steps = args.max_steps
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
        env.eval(max_steps=max_steps, max_search_steps=max_steps)

    env.end(
        ending_time=0.0 if args.headless else 3.0,
        ani_name=world_path.stem,
    )


if __name__ == "__main__":
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
        "--max_steps",
        type=int,
        default=500,
        help="maximum simulation steps",
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
        "--expansion_time",
        type=float,
        default=0.15,
        help="MCTS wall-time budget per replan, matching the original method",
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
        help="draw current robot/target states",
    )
    parser.add_argument(
        "--debug_reward",
        action="store_true",
        default=False,
        help="print reward decomposition from the original navState",
    )
    parser.add_argument(
        "--debug_obstacle",
        action="store_true",
        default=False,
        help="print costmap obstacle hits from the original nodes.any_obs",
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

    args = parser.parse_args()

    config_dir = resolve_config_dir(args.config_path)
    if args.log_path:
        env_path_file = config_dir / f"{args.scenario}_{args.position}_{args.index}.yaml"
    else:
        env_path_file = config_dir / f"{args.scenario}_{args.position}.yaml"

    main(str(env_path_file), args)
