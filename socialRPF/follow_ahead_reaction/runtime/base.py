import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if "--headless" in sys.argv:
    import matplotlib

    matplotlib.use("Agg", force=True)


RUNTIME_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = RUNTIME_DIR.parent
RDA_PLANNER_ROOT = PACKAGE_ROOT.parent
FOLLOW_BENCH_ROOT = RDA_PLANNER_ROOT.parent
ROBOT_PERSON_FOLLOWING_DIR = (
    FOLLOW_BENCH_ROOT / "RDA_planner" / "example" / "robot_person_following"
)
ASSETS_DIR = PACKAGE_ROOT / "assets"
IR_SIM_ROOT = FOLLOW_BENCH_ROOT / "ir-sim"

for candidate in (IR_SIM_ROOT,):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import irsim
from irsim.global_param.path_param import path_manager as pm

from ..models.human_prob_dist import prob_dist
from ..models.RL_interface import RL_model
from ..mcts.nodes import MCTSNode
from ..mcts.search import MCTS
from ..mcts.navi_state import navState
from ..mcts.follow_task_utils import format_distance_tag, model_variant_dir


def inflate_occupancy(grid_map, radius_cells):
    if radius_cells <= 0:
        return grid_map

    occupied = grid_map > 0
    if not np.any(occupied):
        return grid_map

    inflated = occupied.copy()
    height, width = occupied.shape

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue

            src_y_start = max(0, -dy)
            src_y_end = min(height, height - dy)
            src_x_start = max(0, -dx)
            src_x_end = min(width, width - dx)

            dst_y_start = max(0, dy)
            dst_y_end = min(height, height + dy)
            dst_x_start = max(0, dx)
            dst_x_end = min(width, width + dx)

            inflated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] |= occupied[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]

    return np.where(inflated, 100, 0).astype(np.uint8)


def resolve_config_dir(config_path: str) -> Path:
    config_dir = Path(config_path)
    if config_dir.is_absolute():
        return config_dir

    cwd_candidate = Path.cwd() / config_path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    return (ROBOT_PERSON_FOLLOWING_DIR / config_path).resolve()


def build_costmap_from_env(env):
    world = env._world
    resolution = world.buffer_reso if world.buffer_reso > 0 else 0.1
    inflation_radius = 0.55
    inflation_cells = int(np.ceil(inflation_radius / resolution))

    if world.grid_map is not None:
        grid_map = env._generate_gm_for_planning(world.grid_map, env.obstacle_list)
    else:
        width = max(int(np.ceil(world.width / resolution)), 1)
        height = max(int(np.ceil(world.height / resolution)), 1)
        grid_map = np.zeros((height, width), dtype=np.uint8)

        for obstacle in env.obstacle_list:
            try:
                obs_x = obstacle.state[0, 0]
                obs_y = obstacle.state[1, 0]
            except Exception:
                continue

            x_index = int(np.rint((obs_x - world.x_range[0]) / resolution))
            y_index = int(np.rint((obs_y - world.y_range[0]) / resolution))
            radius = max(1, int(0.5 / resolution))

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    mark_x = x_index + dx
                    mark_y = y_index + dy
                    if 0 <= mark_x < width and 0 <= mark_y < height:
                        grid_map[mark_y, mark_x] = 100

    inflated_map = inflate_occupancy(grid_map, inflation_cells)
    costmap = np.where(inflated_map > 0, 100, 0).astype(np.int8)

    params = {
        "map_origin_x": world.x_range[0],
        "map_origin_y": world.y_range[0],
        "map_res": resolution,
        "map_width": costmap.shape[1],
        "map_data": costmap.flatten().tolist(),
        "inflation_radius": inflation_radius,
        "inflation_cells": inflation_cells,
    }
    return params, costmap


class FollowAheadReactionIRSim:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.params = {}
        self.params["robot_vel"] = 0.6
        self.params["robot_vel_fast_lamda"] = 1.5
        self.params["human_vel"] = 0.6
        self.params["dt"] = 0.2
        self.params["gamma"] = 0.9
        self.params["robot_angle"] = 45.0
        self.params["human_angle"] = 10.0
        self.params["safety_params"] = {"r": 0.5, "a": 0.25}
        self.params["reaction_zone_params"] = {"r": 0.8, "a": 0.3}
        self.params["human_acts"] = self.define_human_actions()
        self.params["robot_acts"] = self.define_robot_actions()
        self.params["expansion_time"] = args.expansion_time
        self.params["follow_mode"] = args.position
        self.params["desired_distance"] = float(args.follow_distance)
        self.params["rl_obs_mode"] = args.rl_obs_mode
        self.params["sim"] = False
        self.robot_vel_nominal = self.params["robot_vel"]
        self.robot_fast_nominal = (
            self.params["robot_vel"] * self.params["robot_vel_fast_lamda"]
        )
        self.scene_robot_linear_max = self._scene_robot_linear_max()
        self.scene_target_speed = self._scene_target_speed()

        if self.params["follow_mode"] == "back":
            self.params["safety_params"] = {"r": 0.80, "a": 0.35}
            self.params["reaction_zone_params"] = {"r": 1.05, "a": 0.40}

        costmap_params, self.grid_map = build_costmap_from_env(env)
        self.params.update(costmap_params)
        self.map_height = costmap_params["map_data"].__len__() // costmap_params["map_width"]

        self.stay_bool = True
        self.best_action = None
        self.theta_thr = 20 * np.pi / 180
        self.freq = 5.0
        self.plan_interval = 1.0 / self.freq
        self.plan_elapsed = 0.0

        human_prob_model_dir = ASSETS_DIR / "human_prob.pth"
        if not human_prob_model_dir.exists():
            raise FileNotFoundError(f"Human probability model not found: {human_prob_model_dir}")
        self.human_prob = prob_dist(str(human_prob_model_dir))

        self.human_history = []
        self.human_history_length = 15
        self.history_bootstrap_min_points = 4
        self.last_target_yaw = 0.0
        self.last_target_raw_yaw = 0.0
        self.last_target_used_yaw = 0.0
        self.last_target_speed = 0.0
        self.last_plan_cost = 0.0
        self.state = np.zeros((2, 3), dtype=float)
        self.step_count = 0

        if args.rl_model_override:
            rl_model_dir = Path(args.rl_model_override)
            model_source = "override"
        else:
            rl_model_dir, model_source = self.resolve_rl_model_path(
                args.position,
                args.follow_distance,
                args.rl_obs_mode,
            )
        if not rl_model_dir.exists():
            raise FileNotFoundError(f"RL model not found: {rl_model_dir}")
        self.params["RL_model"] = RL_model()
        self.params["RL_model"].load_model(str(rl_model_dir), policy="a2c")

        if self.args.show_debug:
            print(
                f"[MODEL] follow_mode={self.params['follow_mode']} "
                f"desired_distance={self.params['desired_distance']:.2f} "
                f"rl_obs_mode={self.params['rl_obs_mode']} "
                f"rl_model={rl_model_dir.name} "
                f"model_source={model_source} "
                f"scene_target_speed={self.scene_target_speed:.3f} "
                f"scene_robot_max={self.scene_robot_linear_max:.3f}"
            )

        if self.args.debug_obstacle:
            print(
                f"[MAP] width={self.params['map_width']} "
                f"height={self.map_height} "
                f"res={self.params['map_res']:.3f} "
                f"inflation_m={self.params['inflation_radius']:.2f} "
                f"inflation_cells={self.params['inflation_cells']} "
                f"obstacles={len(self.env.obstacle_list)}"
            )
            for obstacle in self.env.obstacle_list:
                obs_state = obstacle.state.copy().squeeze()
                print(
                    f"[MAP] obstacle={obstacle.name} "
                    f"state={np.round(obs_state, 3).tolist()} "
                    f"shape={obstacle.shape}"
                )

    @staticmethod
    def wrap_to_pi(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def quantize_human_yaw(self, yaw):
        yaw = self.wrap_to_pi(yaw)
        return (np.abs(yaw) // self.theta_thr) * self.theta_thr * np.sign(yaw)

    def _scene_robot_linear_max(self):
        vel_max = np.asarray(getattr(self.env.robot, "vel_max", [[self.robot_fast_nominal]])).astype(float).reshape(-1)
        if vel_max.size == 0:
            return self.robot_fast_nominal
        return max(float(abs(vel_max[0])), self.robot_fast_nominal)

    def _scene_target_speed(self):
        vel_max = np.asarray(getattr(self.env.target, "vel_max", [[self.params["human_vel"]]])).astype(float).reshape(-1)
        if vel_max.size == 0:
            return self.params["human_vel"]

        target_kinematics = getattr(self.env.target, "kinematics", "")
        if target_kinematics in {"diff", "acker"}:
            return max(float(abs(vel_max[0])), self.params["human_vel"])

        return max(float(np.max(np.abs(vel_max[:2]))), self.params["human_vel"])

    def resolve_rl_model_path(self, position, follow_distance, rl_obs_mode="relative_pose"):
        dist_tag = format_distance_tag(follow_distance)
        task_model = ASSETS_DIR / model_variant_dir(rl_obs_mode) / f"follow_value_{position}_{dist_tag}.zip"
        if task_model.exists():
            return task_model, "packaged"

        return task_model, "missing"

    def update_back_motion_params(self, state):
        if self.params.get("follow_mode") != "back":
            return

        distance = float(np.linalg.norm(state[0, :2] - state[1, :2]))
        target_speed = max(self.scene_target_speed, self.last_target_speed, self.params["human_vel"])
        robot_limit = max(self.scene_robot_linear_max, self.robot_fast_nominal)

        # Preserve the original nominal speeds near the desired follow distance,
        # but scale up when the scene's target is much faster and the robot is lagging.
        desired_distance = float(self.params.get("desired_distance", 1.5))
        catch_ratio = np.clip((distance - desired_distance) / 1.5, 0.0, 1.0)

        base_speed_target = min(robot_limit, target_speed)
        robot_vel = self.robot_vel_nominal + catch_ratio * (base_speed_target - self.robot_vel_nominal)

        fast_speed_target = min(robot_limit, max(self.robot_fast_nominal, target_speed + 0.2 * catch_ratio))
        if distance < 1.6:
            fast_speed_target = min(fast_speed_target, max(robot_vel, self.robot_fast_nominal))

        robot_vel = max(float(robot_vel), 1e-3)
        fast_speed_target = max(float(fast_speed_target), robot_vel)

        self.params["human_vel"] = float(target_speed)
        self.params["robot_vel"] = robot_vel
        self.params["robot_vel_fast_lamda"] = fast_speed_target / robot_vel

    def update_side_motion_params(self, state):
        if self.params.get("follow_mode") not in {"left_side", "right_side"}:
            return

        rel_vec = state[0, :2] - state[1, :2]
        human_yaw = state[1, 2]
        heading_vec = np.array([np.cos(human_yaw), np.sin(human_yaw)])
        lateral_vec = np.array([-np.sin(human_yaw), np.cos(human_yaw)])
        lon = float(np.dot(rel_vec, heading_vec))
        lat = float(np.dot(rel_vec, lateral_vec))

        desired_distance = float(self.params.get("desired_distance", 1.5))
        desired_lat = desired_distance if self.params.get("follow_mode") == "left_side" else -desired_distance
        lon_err = lon
        lat_err = lat - desired_lat

        target_speed = max(self.scene_target_speed, self.last_target_speed, self.params["human_vel"])
        robot_limit = max(self.scene_robot_linear_max, self.robot_fast_nominal)

        # For side-follow, the key failure modes are falling behind on the longitudinal axis
        # and then overshooting into the person's front half-plane during turns.
        catch_ratio = np.clip(max(-lon_err, 0.0) / max(1.5 * desired_distance, 1.5), 0.0, 1.0)
        ahead_ratio = np.clip(max(lon_err, 0.0) / max(0.75 * desired_distance, 0.5), 0.0, 1.0)
        lateral_ratio = np.clip(abs(lat_err) / max(desired_distance, 0.5), 0.0, 1.0)

        distance = float(np.linalg.norm(rel_vec))
        if lon_err > 0.05 * desired_distance:
            brake_gain = 0.45 if distance < desired_distance + 0.40 else 0.30
            robot_vel = max(self.robot_vel_nominal, target_speed - brake_gain * ahead_ratio)
            fast_speed_target = max(robot_vel, target_speed - 0.20 * ahead_ratio + 0.06 * lateral_ratio)
        else:
            robot_vel = max(self.robot_vel_nominal, target_speed + 0.12 * catch_ratio)
            fast_speed_target = max(self.robot_fast_nominal, target_speed + 0.55 * catch_ratio + 0.20 * lateral_ratio)

        robot_vel = min(robot_limit, robot_vel)
        fast_speed_target = min(robot_limit, max(fast_speed_target, robot_vel))

        self.params["human_vel"] = float(target_speed)
        self.params["robot_vel"] = float(robot_vel)
        self.params["robot_vel_fast_lamda"] = float(fast_speed_target / max(robot_vel, 1e-3))

    def compute_relative_diagnostics(self, state):
        robot_xy = state[0, :2]
        human_xy = state[1, :2]
        human_yaw = state[1, 2]
        robot_yaw = state[0, 2]
        follow_mode = self.params.get("follow_mode", "front")

        rel_vec = robot_xy - human_xy
        distance = float(np.linalg.norm(rel_vec))
        beta = np.arctan2(rel_vec[1], rel_vec[0])
        mode_offsets = {
            "front": 0.0,
            "back": np.pi,
            "left_side": np.pi / 2.0,
            "right_side": -np.pi / 2.0,
        }
        desired_beta = self.wrap_to_pi(human_yaw + mode_offsets.get(follow_mode, 0.0))
        diff = np.abs(self.wrap_to_pi(desired_beta - beta)) * 180.0 / np.pi

        heading_vec = np.array([np.cos(human_yaw), np.sin(human_yaw)])
        lateral_vec = np.array([-np.sin(human_yaw), np.cos(human_yaw)])

        longitudinal = float(np.dot(rel_vec, heading_vec))
        lateral = float(np.dot(rel_vec, lateral_vec))
        yaw_error = self.wrap_to_pi(robot_yaw - human_yaw) * 180.0 / np.pi

        if longitudinal > 0.3:
            longitudinal_zone = "front"
        elif longitudinal < -0.3:
            longitudinal_zone = "back"
        else:
            longitudinal_zone = "side"

        if lateral > 0.25:
            lateral_zone = "left"
        elif lateral < -0.25:
            lateral_zone = "right"
        else:
            lateral_zone = "center"

        return {
            "distance": distance,
            "diff": diff,
            "follow_mode": follow_mode,
            "longitudinal": longitudinal,
            "lateral": lateral,
            "yaw_error": yaw_error,
            "zone": f"{longitudinal_zone}-{lateral_zone}",
        }

    def get_robot_state(self):
        robot_state = self.env.robot.state.copy().squeeze()
        robot_state[2] = self.wrap_to_pi(robot_state[2])
        return robot_state

    def get_human_state(self):
        target_velocity = self.env.target.velocity
        target_pose = self.env.target.state.copy().squeeze()
        raw_yaw = self.wrap_to_pi(float(target_pose[2]))
        self.last_target_raw_yaw = raw_yaw
        target_kinematics = getattr(self.env.target, "kinematics", "")

        speed = float(
            np.sqrt(target_velocity[0][0] ** 2 + target_velocity[1][0] ** 2)
        )
        self.last_target_speed = speed

        if target_kinematics in {"diff", "acker"}:
            yaw = raw_yaw
            self.last_target_yaw = yaw
        else:
            if speed > 1e-3:
                observed_yaw = np.arctan2(target_velocity[1][0], target_velocity[0][0])
                max_delta = self.params["human_angle"] * np.pi / 180.0
                delta = self.wrap_to_pi(observed_yaw - self.last_target_yaw)
                delta = np.clip(delta, -max_delta, max_delta)
                yaw = self.wrap_to_pi(self.last_target_yaw + delta)
                self.last_target_yaw = yaw
            else:
                yaw = self.last_target_yaw

        target_pose[2] = self.quantize_human_yaw(yaw)
        self.last_target_used_yaw = float(target_pose[2])
        return target_pose

    def get_human_history_point(self):
        target_pose = self.env.target.state.copy().squeeze()
        return target_pose[:2].tolist()

    def update_human_history(self):
        self.human_history.append(self.get_human_history_point())
        if len(self.human_history) <= self.human_history_length:
            return False

        self.human_history.pop(0)
        return True

    def build_bootstrap_history(self):
        if len(self.human_history) < self.history_bootstrap_min_points:
            return None

        history = np.asarray(self.human_history, dtype=float)
        if len(history) >= 2:
            step_vec = np.mean(np.diff(history, axis=0), axis=0)
        else:
            step_vec = np.zeros(2, dtype=float)

        if np.linalg.norm(step_vec) < 1e-6:
            target_velocity = np.asarray(self.env.target.velocity, dtype=float).reshape(-1)
            if target_velocity.size >= 2:
                step_vec = target_velocity[:2] * self.plan_interval

        pad_count = self.human_history_length - len(history)
        if pad_count <= 0:
            return history.tolist()[-self.human_history_length :]

        anchor = history[0]
        padding = [anchor - step_vec * (pad_count - idx) for idx in range(pad_count)]
        bootstrap_history = np.vstack([padding, history])
        return bootstrap_history.tolist()

    def get_planning_history(self, history_ready):
        if history_ready:
            return list(self.human_history)

        if self.params.get("follow_mode") in {"back", "left_side", "right_side"}:
            return self.build_bootstrap_history()

        return None

    def current_state(self):
        state = np.zeros((2, 3), dtype=float)
        state[0, :] = self.get_robot_state()
        state[1, :] = self.get_human_state()
        self.state = state
        return state

    def define_human_actions(self):
        return {
            0: "left",
            1: "right",
            2: "straight",
        }

    def define_robot_actions(self):
        return {
            0: "fast_left",
            1: "fast_right",
            2: "fast_straight",
            3: "left",
            4: "right",
            5: "straight",
        }

    def expand_tree(self, state, human_prob=None):
        if not self.stay(state):
            if self.args.show_debug:
                print(
                    f"[PLAN] step={self.step_count} start MCTS "
                    f"dist={np.linalg.norm(state[0, :2] - state[1, :2]):.3f} "
                    f"history={len(self.human_history)}"
                )
            nav_state = navState(params=self.params, state=state, next_to_move=0)
            node_human = MCTSNode(state=nav_state, params=self.params, parent=None)
            mcts = MCTS(node_human, human_prob)
            best_node = mcts.tree_expantion(
                time.time() + self.params["expansion_time"]
            )
            if best_node:
                if self.args.show_debug:
                    print(f"[PLAN] step={self.step_count} action={best_node.action}")
                return best_node.action
            if self.args.show_debug:
                print(f"[PLAN] step={self.step_count} action=stop (no best node)")
            return "stop"

        if self.args.show_debug:
            print(
                f"[WAIT] step={self.step_count} "
                f"dist={np.linalg.norm(state[0, :2] - state[1, :2]):.3f} > 1.5"
            )
        return None

    def stay(self, state):
        follow_mode = self.params.get("follow_mode", "front")
        if follow_mode != "front":
            return False

        if self.stay_bool:
            distance = np.linalg.norm(state[0, :2] - state[1, :2])
            if distance > 1.5:
                return True

            self.stay_bool = False
            return False

        return False

    def command_from_action(self):
        action = self.best_action
        if self.params.get("follow_mode") == "back" and self.state is not None:
            distance = np.linalg.norm(self.state[0, :2] - self.state[1, :2])
            if distance < 1.2:
                action = "stop"
            elif distance < 1.45 and action in {"left", "right", "fast_left", "fast_right"}:
                action = "stop"
            elif distance < 1.55 and action in {"fast_straight", "fast_left", "fast_right"}:
                action = action.replace("fast_", "")

        linear_vel = self.params["robot_vel"]
        if action in {"fast_straight", "fast_right", "fast_left"}:
            linear_vel *= self.params["robot_vel_fast_lamda"]

        angular_vel = self.params["robot_angle"] * np.pi / 180
        if action in {"straight", "fast_straight"}:
            angular_vel = 0.0
        elif action in {"right", "fast_right"}:
            angular_vel *= -1.0
        elif action in {"stop", None}:
            linear_vel = 0.0
            angular_vel = 0.0

        return np.array([[linear_vel], [angular_vel]])

    def maybe_plan(self, state):
        self.plan_elapsed += self.env.step_time
        if self.plan_elapsed + 1e-9 < self.plan_interval:
            return 0.0

        self.plan_elapsed -= self.plan_interval
        history_ready = self.update_human_history()
        diag = self.compute_relative_diagnostics(state)
        if self.args.show_debug:
            print(
                f"[STATE] step={self.step_count} "
                f"robot={np.round(state[0], 3).tolist()} "
                f"target={np.round(state[1], 3).tolist()} "
                f"dist={diag['distance']:.3f} "
                f"diff={diag['diff']:.3f} "
                f"lon={diag['longitudinal']:.3f} "
                f"lat={diag['lateral']:.3f} "
                f"yaw_err={diag['yaw_error']:.3f} "
                f"zone={diag['zone']} "
                f"history={len(self.human_history)}/{self.human_history_length}"
            )
            print(
                f"[TARGET] step={self.step_count} "
                f"raw_pose={np.round(self.env.target.state.copy().squeeze(), 3).tolist()} "
                f"speed={self.last_target_speed:.3f} "
                f"raw_yaw={self.last_target_raw_yaw * 180.0 / np.pi:.3f} "
                f"used_yaw={self.last_target_used_yaw * 180.0 / np.pi:.3f}"
            )
            if self.params.get("follow_mode") in {"back", "left_side", "right_side"}:
                print(
                    f"[DYN] step={self.step_count} "
                    f"human_vel={self.params['human_vel']:.3f} "
                    f"robot_vel={self.params['robot_vel']:.3f} "
                    f"robot_fast={self.params['robot_vel'] * self.params['robot_vel_fast_lamda']:.3f}"
                )
        planning_history = self.get_planning_history(history_ready)
        if planning_history is None:
            if self.args.show_debug:
                print(
                    f"[HISTORY] step={self.step_count} "
                    f"waiting for history warmup"
                )
            return 0.0
        if self.args.show_debug and not history_ready:
            print(
                f"[HISTORY] step={self.step_count} "
                f"bootstrap_history={len(planning_history)}/{self.human_history_length}"
            )

        human_prob = self.human_prob.forward(planning_history)
        if self.args.show_debug:
            print(
                f"[PROB] step={self.step_count} "
                f"left={human_prob['left']:.3f} "
                f"straight={human_prob['straight']:.3f} "
                f"right={human_prob['right']:.3f}"
            )

        plan_start = time.time()
        self.best_action = self.expand_tree(state, human_prob)
        self.last_plan_cost = time.time() - plan_start
        if self.args.show_debug:
            print(
                f"[CTRL] step={self.step_count} "
                f"best_action={self.best_action} "
                f"plan_cost={self.last_plan_cost:.3f}s"
            )
        return self.last_plan_cost

    def draw_debug(self, state):
        if not self.args.show_debug:
            return

        robot_point = state[0, :2][:, np.newaxis]
        human_point = state[1, :2][:, np.newaxis]
        self.env.draw_points(robot_point, s=60, c="r", refresh=True)
        self.env.draw_points(human_point, s=60, c="b", refresh=False)

    def step(self):
        self.step_count += 1
        state = self.current_state()
        self.update_back_motion_params(state)
        self.update_side_motion_params(state)
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
    controller = FollowAheadReactionIRSim(env, args)

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
        default="../../follow_ahead/config",
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
        "--no_save_animation",
        action="store_false",
        dest="save_animation",
        help="disable gif generation at the end of the run",
    )
    parser.set_defaults(save_animation=True)

    args = parser.parse_args()

    config_dir = resolve_config_dir(args.config_path)
    if args.log_path:
        env_path_file = config_dir / f"{args.scenario}_{args.position}_{args.index}.yaml"
    else:
        env_path_file = config_dir / f"{args.scenario}_{args.position}.yaml"

    main(str(env_path_file), args)
