"""
BSO-HFC differential-drive person following on the Follow-Bench stack.

Planning uses :class:`BSOHFCPlanner` with the same task-target construction as in this
module. Step visualization follows ``dwa_improved_planner_diff.py`` by default (black
reference goal polyline, red start marker, red MPC horizon). Optional
``--show_planner_debug`` restores the full local-map / hybrid-A* / spline overlay.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from collections import deque, namedtuple
from dataclasses import dataclass, field
from datetime import datetime

if "--headless" in sys.argv:
    os.environ["MPLBACKEND"] = "Agg"

import irsim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Repo root contains the RDA_planner and traj_predictor packages (example/robot_person_following -> ../..).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from BSO_HFC_planner import BSOHFCPlanner, DiscSpec, load_bso_hfc_config

try:
    from traj_predictor import get_predictor
except ImportError:
    get_predictor = None

from global_params import get_predicted_target_pose, predict_trajectory, traj_predictors, update_trajectory

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

### for large-scale test, shutdown visualization (uncomment like dwa_improved_planner_diff) ###
# os.environ["MPLBACKEND"] = "Agg"  # must be before any pyplot use if enabled
# matplotlib.use("Agg")
# plt.ioff()
### for large-scale test, shutdown visualization ###

robot = namedtuple("robot", "shape radius length width min_speed max_speed max_acce kinematics")


@dataclass
class FollowBenchState:
    target_history_maxlen: int
    mode: str = "TRACK_VISIBLE"
    lost_frame_count: int = 0
    reacquire_frame_count: int = 0
    last_seen_target_pose_world: np.ndarray | None = None
    search_reference_point_world: np.ndarray | None = None
    search_direction_world: np.ndarray | None = None
    search_initialized: bool = False
    last_seen_target_state_history: deque = field(init=False)

    def __post_init__(self) -> None:
        # Bounded target history for guidance and recovery.
        self.last_seen_target_state_history = deque(maxlen=max(int(self.target_history_maxlen), 1))


@dataclass
class TaskTarget:
    planning_target_pose_world: np.ndarray
    planning_target_traj_world_xy: np.ndarray
    clear_goal_pose_world: np.ndarray | None
    clear_goal_radius: float
    occupied_discs_world: list
    display_target_pose_world: np.ndarray | None
    mode_label: str
    desired_follow_pose_world: np.ndarray | None


@dataclass
class TimingMonitor:
    window: int
    total_history_ms: deque = field(init=False)

    def __post_init__(self) -> None:
        self.total_history_ms = deque(maxlen=max(int(self.window), 1))

    def extract_timing_ms(self, info: dict) -> dict:
        if not isinstance(info, dict):
            return {
                "frontend_ms": 0.0,
                "backend_ms": 0.0,
                "mpc_ms": 0.0,
                "total_ms": 0.0,
            }

        stage_timing_ms = info.get("stage_timing_ms") if isinstance(info.get("stage_timing_ms"), dict) else {}
        frontend_ms = float(stage_timing_ms.get("hybrid_astar", 0.0))
        backend_ms = float(stage_timing_ms.get("bspline", 0.0))
        mpc_ms = float(stage_timing_ms.get("mpc", 0.0))
        total_ms = frontend_ms + backend_ms + mpc_ms
        return {
            "frontend_ms": frontend_ms,
            "backend_ms": backend_ms,
            "mpc_ms": mpc_ms,
            "total_ms": total_ms,
        }

    def format(self, info: dict) -> str:
        timing_ms = self.extract_timing_ms(info)
        self.total_history_ms.append(timing_ms["total_ms"])
        avg_total_ms = float(np.mean(self.total_history_ms)) if len(self.total_history_ms) > 0 else 0.0

        total_ms = timing_ms["total_ms"]
        freq_hz = 1000.0 / total_ms if total_ms > 1e-6 else 0.0
        avg_freq_hz = 1000.0 / avg_total_ms if avg_total_ms > 1e-6 else 0.0
        return (
            f"front-end={timing_ms['frontend_ms']:.2f} ms, "
            f"back-end={timing_ms['backend_ms']:.2f} ms, "
            f"mpc={timing_ms['mpc_ms']:.2f} ms, "
            f"total={total_ms:.2f} ms ({freq_hz:.2f} Hz), "
            f"avg{self.window}={avg_total_ms:.2f} ms ({avg_freq_hz:.2f} Hz)"
        )


def coerce_pose_vector(pose) -> np.ndarray:
    arr = np.asarray(pose, dtype=float).reshape(-1)
    if arr.size >= 3:
        return arr[:3].copy()
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0], dtype=float)
    raise ValueError("pose must contain at least x and y")


def history_to_array(history: deque) -> np.ndarray:
    if len(history) == 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray([coerce_pose_vector(pose) for pose in history], dtype=float)


def normalize_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    direction = np.asarray(vec, dtype=float).reshape(-1)[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.asarray(fallback, dtype=float).reshape(-1)[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return np.array([1.0, 0.0], dtype=float)
    return direction / norm


def get_target_person_state(env, yaw_target_list):
    target_person_vel = env.target.velocity
    target_person_pose = env.target.state

    v_target = np.sqrt(target_person_vel[0][0] ** 2 + target_person_vel[1][0] ** 2)
    yaw_target = np.atan2(target_person_vel[1][0], target_person_vel[0][0])
    if v_target < 0.01 and len(yaw_target_list) > 0:
        yaw_target = yaw_target_list[-1]
    yaw_target_list.append(yaw_target)

    if len(yaw_target_list) > 1:
        w_target = np.gradient(yaw_target_list, env.step_time)[-1]
    else:
        w_target = 0.0

    target_person_pose[2] = yaw_target
    target_person_vel = np.array([[v_target], [w_target]])
    return target_person_vel, target_person_pose


def initialize_followbench_state(target_history_maxlen: int) -> FollowBenchState:
    return FollowBenchState(target_history_maxlen=target_history_maxlen)


def initialize_search_history(traj_predictor_params: dict, prediction_horizon: float) -> dict:
    # Match the benchmark predictor history format.
    history_length = int(traj_predictor_params["history_length"])
    return {
        "updated_num": 0,
        "history_length": history_length,
        "dt": float(traj_predictor_params["dt"]),
        "prediction_horizon": float(prediction_horizon),
        "robot": deque(maxlen=history_length),
        "target": deque(maxlen=history_length),
        "humans": {},
    }


def update_target_history(state: FollowBenchState, target_person_pose) -> None:
    # Cache the last visible target state for side-follow and recovery.
    pose = coerce_pose_vector(target_person_pose)
    state.last_seen_target_pose_world = pose.copy()
    state.last_seen_target_state_history.append(pose.copy())
    state.search_reference_point_world = None
    state.search_initialized = False


def update_tracking_mode(
    state: FollowBenchState,
    target_visible: bool,
    lost_confirm_steps: int,
    reacquire_confirm_steps: int,
) -> str:
    # Visibility state machine: track, confirm loss, then search.
    if target_visible:
        state.lost_frame_count = 0
        if state.mode in {"SEARCH", "LOST_CONFIRM"}:
            state.reacquire_frame_count += 1
            if state.reacquire_frame_count >= max(int(reacquire_confirm_steps), 1):
                state.mode = "TRACK_VISIBLE"
            else:
                state.mode = "REACQUIRE_TRANSITION"
        else:
            state.reacquire_frame_count = 0
            state.mode = "TRACK_VISIBLE"
        return state.mode

    state.reacquire_frame_count = 0
    state.lost_frame_count += 1
    if state.lost_frame_count >= max(int(lost_confirm_steps), 1):
        state.mode = "SEARCH"
    else:
        state.mode = "LOST_CONFIRM"
    return state.mode


def compute_smoothed_headings(real_target_state_history: np.ndarray, fallback_yaw: float, window: int = 5) -> np.ndarray:
    # Smooth the target heading from history to reduce corner flips.
    history = np.asarray(real_target_state_history, dtype=float)
    if history.size == 0:
        return np.empty((0,), dtype=float)

    positions = history[:, :2]
    fallback_dir = np.array([np.cos(float(fallback_yaw)), np.sin(float(fallback_yaw))], dtype=float)
    headings = []
    for i in range(len(positions)):
        start = max(0, i - max(int(window), 1) + 1)
        diffs = np.diff(positions[start : i + 1], axis=0)
        if diffs.size == 0:
            direction = fallback_dir
        else:
            direction = normalize_vector(np.sum(diffs, axis=0), fallback_dir)
        headings.append(float(np.arctan2(direction[1], direction[0])))
    return np.asarray(headings, dtype=float)


def compute_side_follow_poses(target_xy: np.ndarray, heading: float, side_mode: str, distance: float) -> tuple[np.ndarray, np.ndarray]:
    # Convert side-follow into back-following a virtual target.
    theta = float(heading)
    e_f = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    e_l = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    side_dir = e_l if side_mode == "left_side" else -e_l
    desired_xy = np.asarray(target_xy, dtype=float).reshape(-1)[:2] + float(distance) * side_dir
    virtual_xy = desired_xy + float(distance) * e_f
    desired_pose = np.array([desired_xy[0], desired_xy[1], theta], dtype=float)
    virtual_pose = np.array([virtual_xy[0], virtual_xy[1], theta], dtype=float)
    return desired_pose, virtual_pose


def build_side_follow_history(real_target_state_history: np.ndarray, side_mode: str, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Rebuild a task-level virtual trajectory for guidance.
    history = np.asarray(real_target_state_history, dtype=float)
    if history.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    fallback_yaw = float(history[-1, 2]) if history.shape[1] >= 3 else 0.0
    headings = compute_smoothed_headings(history, fallback_yaw)
    desired_points = []
    virtual_points = []
    for pose, heading in zip(history, headings):
        desired_pose, virtual_pose = compute_side_follow_poses(pose[:2], float(heading), side_mode, distance)
        desired_points.append(desired_pose[:2])
        virtual_points.append(virtual_pose[:2])
    return np.asarray(desired_points, dtype=float), np.asarray(virtual_points, dtype=float), headings


def compute_back_follow_pose(target_pose_world, robot_pose_world, distance: float) -> np.ndarray:
    # Build a display-only back-follow target behind the real person.
    target_pose = coerce_pose_vector(target_pose_world)
    robot_pose = coerce_pose_vector(robot_pose_world)
    direction = normalize_vector(robot_pose[:2] - target_pose[:2], np.array([np.cos(target_pose[2]), np.sin(target_pose[2])], dtype=float))
    desired_xy = target_pose[:2] + float(distance) * direction
    yaw = float(np.arctan2(target_pose[1] - desired_xy[1], target_pose[0] - desired_xy[0]))
    return np.array([desired_xy[0], desired_xy[1], yaw], dtype=float)


def build_back_task_target(robot_pose_world, real_target_pose, real_target_state_history: np.ndarray, target_radius: float, distance: float) -> TaskTarget:
    target_pose = coerce_pose_vector(real_target_pose)
    desired_pose = compute_back_follow_pose(target_pose, robot_pose_world, distance)
    history_xy = real_target_state_history[:, :2] if real_target_state_history.size > 0 else target_pose[:2].reshape(1, 2)
    return TaskTarget(
        planning_target_pose_world=target_pose,
        planning_target_traj_world_xy=np.asarray(history_xy, dtype=float),
        clear_goal_pose_world=target_pose.copy(),
        clear_goal_radius=float(target_radius),
        occupied_discs_world=[],
        display_target_pose_world=desired_pose.copy(),
        mode_label="back",
        desired_follow_pose_world=desired_pose.copy(),
    )


def build_side_task_target(
    real_target_pose,
    real_target_state_history: np.ndarray,
    side_mode: str,
    distance: float,
    protected_radius: float,
) -> TaskTarget:
    target_pose = coerce_pose_vector(real_target_pose)
    history = real_target_state_history if real_target_state_history.size > 0 else target_pose.reshape(1, 3)
    desired_traj, virtual_traj, headings = build_side_follow_history(history, side_mode, distance)
    current_heading = float(headings[-1]) if len(headings) > 0 else float(target_pose[2])
    desired_pose, virtual_pose = compute_side_follow_poses(target_pose[:2], current_heading, side_mode, distance)

    if virtual_traj.size == 0:
        virtual_traj = virtual_pose[:2].reshape(1, 2)

    occupied_discs_world = [
        DiscSpec(
            center_world_xy=target_pose[:2].copy(),
            radius=max(float(protected_radius), 0.0),
        )
    ]
    return TaskTarget(
        planning_target_pose_world=virtual_pose,
        planning_target_traj_world_xy=virtual_traj,
        clear_goal_pose_world=None,
        clear_goal_radius=0.0,
        occupied_discs_world=occupied_discs_world,
        display_target_pose_world=virtual_pose.copy(),
        mode_label=side_mode,
        desired_follow_pose_world=desired_pose.copy(),
    )


def compute_search_reference_point(
    robot_pose,
    state: FollowBenchState,
    traj_predictor,
    his_traj: dict | None,
    search_step: float,
) -> np.ndarray:
    # Build the benchmark-style search reference point.
    robot_pose_vec = coerce_pose_vector(robot_pose)
    robot_xy = robot_pose_vec[:2]
    default_dir = np.array([np.cos(robot_pose_vec[2]), np.sin(robot_pose_vec[2])], dtype=float)

    predicted_xy = None
    if traj_predictor is not None and his_traj is not None:
        history_length = int(his_traj["history_length"])
        if len(his_traj["target"]) >= history_length and len(his_traj["robot"]) >= history_length:
            try:
                target_future_traj, _ = predict_trajectory(traj_predictor, his_traj)
                predicted_target_pose = get_predicted_target_pose(target_future_traj)
                predicted_xy = np.asarray(predicted_target_pose, dtype=float).reshape(-1)[:2]
            except Exception:
                predicted_xy = None

    if predicted_xy is not None:
        direction = normalize_vector(predicted_xy - robot_xy, default_dir)
        state.search_direction_world = direction
        state.search_reference_point_world = predicted_xy.copy()
        state.search_initialized = True
        return predicted_xy.copy()

    if not state.search_initialized:
        history = history_to_array(state.last_seen_target_state_history)
        if len(history) >= 2:
            init_dir = history[-1, :2] - history[-2, :2]
        elif state.last_seen_target_pose_world is not None:
            init_dir = np.asarray(state.last_seen_target_pose_world, dtype=float).reshape(-1)[:2] - robot_xy
        else:
            init_dir = default_dir
        state.search_direction_world = normalize_vector(init_dir, default_dir)
        state.search_initialized = True

    search_direction = normalize_vector(state.search_direction_world, default_dir)
    anchor_xy = robot_xy if state.last_seen_target_pose_world is None else np.asarray(state.last_seen_target_pose_world, dtype=float).reshape(-1)[:2]
    search_xy = anchor_xy + float(max(search_step, 1e-3)) * search_direction
    state.search_direction_world = search_direction
    state.search_reference_point_world = search_xy.copy()
    return search_xy


def build_search_task_target(robot_pose, state: FollowBenchState, search_reference_point_world) -> TaskTarget:
    # Turn the search reference point into a temporary planning target.
    robot_pose_vec = coerce_pose_vector(robot_pose)
    search_xy = np.asarray(search_reference_point_world, dtype=float).reshape(-1)[:2]
    direction = normalize_vector(
        search_xy - robot_pose_vec[:2],
        np.array([np.cos(robot_pose_vec[2]), np.sin(robot_pose_vec[2])], dtype=float),
    )
    yaw = float(np.arctan2(direction[1], direction[0]))
    planning_target_pose_world = np.array([search_xy[0], search_xy[1], yaw], dtype=float)

    if state.last_seen_target_pose_world is not None:
        anchor_xy = np.asarray(state.last_seen_target_pose_world, dtype=float).reshape(-1)[:2]
    else:
        anchor_xy = robot_pose_vec[:2]

    if np.linalg.norm(search_xy - anchor_xy) <= 1e-6:
        planning_traj = search_xy.reshape(1, 2)
    else:
        planning_traj = np.vstack((anchor_xy.reshape(1, 2), search_xy.reshape(1, 2)))

    return TaskTarget(
        planning_target_pose_world=planning_target_pose_world,
        planning_target_traj_world_xy=planning_traj,
        clear_goal_pose_world=None,
        clear_goal_radius=0.0,
        occupied_discs_world=[],
        display_target_pose_world=planning_target_pose_world.copy(),
        mode_label="search",
        desired_follow_pose_world=None,
    )


def build_goal_traj_list_from_task_target(task_target: TaskTarget | None) -> list[np.ndarray]:
    """World-frame goal polyline as a list of (3, 1) poses, matching ``dwa_improved_planner_diff`` / ``getRobotGoalTraj``."""
    if task_target is None:
        return []
    end_pose = coerce_pose_vector(task_target.planning_target_pose_world)
    xy = np.asarray(task_target.planning_target_traj_world_xy, dtype=float)
    if xy.size == 0:
        return [np.array([[end_pose[0]], [end_pose[1]], [end_pose[2]]], dtype=float)]
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)
    xy = xy[:, :2]
    diffs = np.diff(xy, axis=0)
    if diffs.size == 0:
        yaws = np.array([float(end_pose[2])], dtype=float)
    else:
        segment_yaws = np.arctan2(diffs[:, 1], diffs[:, 0])
        yaws = np.append(segment_yaws, float(end_pose[2]))
    goal_traj: list[np.ndarray] = []
    for i in range(xy.shape[0]):
        goal_traj.append(np.array([[xy[i, 0]], [xy[i, 1]], [yaws[i]]], dtype=float))
    return goal_traj


def make_idle_info(distance: float, target_traj_world_xy: np.ndarray | None = None) -> dict:
    target_traj_world_xy = np.asarray(target_traj_world_xy, dtype=float) if target_traj_world_xy is not None else np.empty((0, 2), dtype=float)
    target_plot = np.empty((2, 0), dtype=float) if target_traj_world_xy.size == 0 else target_traj_world_xy[:, :2].T
    empty_path = np.empty((2, 0), dtype=float)
    zero_xy = np.zeros((2,), dtype=float)
    return {
        "arrive": False,
        "planning_success": False,
        "stage_timing_ms": {
            "hybrid_astar": 0.0,
            "hybrid_astar_search": 0.0,
            "fallback": 0.0,
            "bspline": 0.0,
            "mpc": 0.0,
        },
        "d_current": 0.0,
        "d_desired": float(distance),
        "d_j": 0.0,
        "v_ref": 0.0,
        "delta_t": 0.0,
        "fallback_used": False,
        "planning_target_world": zero_xy.copy(),
        "clear_goal_world": None,
        "occupied_discs_world": [],
        "target_raw_local": zero_xy.copy(),
        "local_goal_raw_local": zero_xy.copy(),
        "local_goal_clamped_local": zero_xy.copy(),
        "local_goal_projected_local": zero_xy.copy(),
        "target_outside_local_map": False,
        "target_raw_world": zero_xy.copy(),
        "local_goal_raw_world": zero_xy.copy(),
        "local_goal_clamped_world": zero_xy.copy(),
        "local_goal_projected_world": zero_xy.copy(),
        "local_map_resolution": 0.0,
        "local_map_shape": np.zeros((2,), dtype=int),
        "local_map_extent_m": zero_xy.copy(),
        "local_map_window_size_m_requested": None,
        "local_map_square_path_list": empty_path,
        "local_map_occupancy_points": empty_path,
        "observable_circle_path_list": empty_path,
        "observable_radius": 0.0,
        "local_map_edt_points": empty_path,
        "local_map_edt_values": np.empty((0,), dtype=float),
        "show_edt": False,
        "target_traj_path_list": target_plot,
        "hybrid_astar_path_list": empty_path,
        "bspline_path_list": empty_path,
        "mpc_path_list": empty_path,
    }


def draw_local_map_edt(env, info: dict) -> None:
    points = np.asarray(info.get("local_map_edt_points", np.empty((2, 0), dtype=float)), dtype=float)
    values = np.asarray(info.get("local_map_edt_values", np.empty((0,), dtype=float)), dtype=float).reshape(-1)
    if points.size == 0 or values.size == 0 or points.shape[0] != 2:
        return

    vmax = float(np.max(values)) if values.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    norm_values = np.clip(values / vmax, 0.0, 1.0)
    bin_count = 20
    bin_ids = np.clip((norm_values * (bin_count - 1)).astype(int), 0, bin_count - 1)
    base_cmap = plt.cm.YlOrBr(np.linspace(0.15, 0.85, bin_count))
    light_cmap = []
    for rgba in base_cmap:
        rgb = np.asarray(rgba[:3], dtype=float)
        rgb = 0.45 * rgb + 0.55 * np.ones((3,), dtype=float)
        light_cmap.append(matplotlib.colors.to_hex(np.clip(rgb, 0.0, 1.0)))

    for idx in range(bin_count):
        mask = bin_ids == idx
        if np.any(mask):
            env.draw_points(points[:, mask], s=6, c=light_cmap[idx], refresh=True)


def draw_planner_debug_overlay(env, info: dict, task_target: TaskTarget | None, real_target_pose_world) -> None:
    """Full BSO-HFC debug drawing (local map, EDT, hybrid-A*, spline, markers). Not used in the default DWA-like path."""
    # Draw map layers first so trajectories and task points remain readable on top.
    if info.get("local_map_occupancy_points", np.empty((2, 0), dtype=float)).size > 0:
        env.draw_points(info["local_map_occupancy_points"], s=6, c="lightgray", refresh=True)

    if info.get("show_edt", False):
        draw_local_map_edt(env, info)

    if info.get("observable_circle_path_list", np.empty((2, 0), dtype=float)).size > 0:
        env.draw_trajectory(info["observable_circle_path_list"], "-.", color="dimgray", linewidth=1.4, alpha=0.85, refresh=True)

    if info.get("local_map_square_path_list", np.empty((2, 0), dtype=float)).size > 0:
        env.draw_trajectory(info["local_map_square_path_list"], "--", color="gray", linewidth=1.2, alpha=0.8, refresh=True)

    if task_target is not None and len(task_target.occupied_discs_world) > 0:
        occupied_points = np.asarray(
            [np.asarray(disc.center_world_xy, dtype=float).reshape(-1)[:2] for disc in task_target.occupied_discs_world],
            dtype=float,
        ).T
        env.draw_points(occupied_points, s=25, c="m", refresh=True)

    local_goal_world = np.asarray(info.get("local_goal_projected_world", []), dtype=float).reshape(-1)
    if local_goal_world.size >= 2:
        env.draw_points(local_goal_world[:2].reshape(2, 1), s=45, c="crimson", refresh=True)

    if info["target_traj_path_list"].size > 0:
        env.draw_trajectory(
            info["target_traj_path_list"],
            ":",
            color="mediumpurple",
            linewidth=1.6,
            alpha=0.85,
            refresh=True,
        )

    if info["bspline_path_list"].size > 0:
        env.draw_trajectory(
            info["bspline_path_list"],
            "-",
            color="teal",
            linewidth=3.0,
            marker="o",
            markersize=2.0,
            alpha=0.82,
            refresh=True,
        )

    if info["hybrid_astar_path_list"].size > 0:
        env.draw_trajectory(
            info["hybrid_astar_path_list"],
            "-.",
            color="darkorange",
            linewidth=2.8,
            marker="s",
            markersize=4.0,
            alpha=0.98,
            refresh=True,
        )

    if info["mpc_path_list"].size > 0:
        env.draw_trajectory(
            info["mpc_path_list"],
            "-",
            color="crimson",
            linewidth=2.2,
            alpha=0.95,
            refresh=True,
        )

    # Orange is the real person; red is the task/planning target; gold is the desired follow pose.
    if real_target_pose_world is not None:
        real_target_xy = np.asarray(real_target_pose_world, dtype=float).reshape(-1)[:2].reshape(2, 1)
        env.draw_points(real_target_xy, s=40, c="darkorange", refresh=True)

    if task_target is not None and task_target.desired_follow_pose_world is not None:
        desired_target_xy = np.asarray(task_target.desired_follow_pose_world, dtype=float).reshape(-1)[:2].reshape(2, 1)
        env.draw_points(desired_target_xy, s=62, c="gold", refresh=True)

    if task_target is not None and task_target.display_target_pose_world is not None:
        planning_target_xy = np.asarray(task_target.display_target_pose_world, dtype=float).reshape(-1)[:2].reshape(2, 1)
        env.draw_points(planning_target_xy, s=86, c="r", refresh=True)


def main(world_name, args):
    # The benchmark layer selects the task target; the core stays pure BSO-HFC.
    if args.position not in {"back", "left_side", "right_side"}:
        raise ValueError("BSO-HFC benchmark integration supports --position back/left_side/right_side.")

    if args.log_path != "":
        eval_dir = os.path.join(args.log_path, os.path.basename(world_name).replace(".yaml", ""))
    else:
        eval_dir = ""

    enable_display = not args.headless
    enable_step_visuals = not args.headless

    env = irsim.make(
        world_name=world_name,
        save_ani=not args.headless,
        display=enable_display,
        full=False,
        eval_dir=eval_dir,
    )

    local_map = getattr(getattr(env.robot, "lidar", None), "local_map", None)
    if local_map is None:
        raise RuntimeError("env.robot.lidar.local_map is required. Please enable lidar 'build_map: True' in the scenario yaml.")

    robot_info = env.get_robot_info()
    vel_min, vel_max = env.robot.get_vel_range()
    max_acce = robot_info.acce.flatten().tolist()
    robot_tuple = robot(
        robot_info.shape,
        env.robot.radius,
        env.robot.length,
        env.robot.width,
        vel_min.flatten().tolist(),
        vel_max.flatten().tolist(),
        max_acce,
        robot_info.kinematics,
    )

    planner_cfg = load_bso_hfc_config(
        robot_tuple,
        env.step_time,
        d_desired=args.distance,
        target_radius=float(getattr(env.target, "radius", 0.0)),
    )
    bso_hfc_planner = BSOHFCPlanner(planner_cfg)
    timing_monitor = TimingMonitor(window=args.timing_avg_window)

    traj_predictor = None
    search_history = None
    if get_predictor is not None:
        traj_predictor_params = dict(traj_predictors[args.traj_predictor])
        traj_predictor_params["dt"] = env.step_time
        traj_predictor = get_predictor(traj_predictor_params["name"], traj_predictor_params)
        search_history = initialize_search_history(traj_predictor_params, prediction_horizon=2.0)

    yaw_target_list = []
    followbench_state = initialize_followbench_state(args.target_history_maxlen)

    if len(env.human_list) == 0:
        max_steps = args.min_steps
    else:
        max_steps = args.min_steps * 2
    max_search_steps = int(args.min_steps * 0.5)

    pbar = tqdm(range(max_steps))
    for i in pbar:
        target_person_vel, target_person_pose = get_target_person_state(env, yaw_target_list)
        robot_vel = env.robot.velocity
        robot_pose = env.robot.state
        target_visible = bool(getattr(env, "check_target_visible", True))

        previous_mode = followbench_state.mode
        mode = update_tracking_mode(
            followbench_state,
            target_visible,
            args.lost_confirm_steps,
            args.reacquire_confirm_steps,
        )

        if target_visible:
            update_target_history(followbench_state, target_person_pose)
            if search_history is not None:
                update_trajectory(env, search_history)

        if (previous_mode != "SEARCH" and mode == "SEARCH") or (previous_mode == "SEARCH" and mode != "SEARCH"):
            bso_hfc_planner.reset()

        task_target = None
        info = make_idle_info(args.distance)
        opt_vel = np.zeros((2, 1), dtype=float)
        alg_cost_t = 0.0
        hold_error = 0.0
        should_hold = False

        real_target_pose_for_display = coerce_pose_vector(target_person_pose) if target_visible else followbench_state.last_seen_target_pose_world
        real_target_history = history_to_array(followbench_state.last_seen_target_state_history)
        target_radius = float(getattr(env.target, "radius", 0.0))

        # --- Build task target (same semantics as before; control runs after DWA-style goal preview.) ---
        if mode in {"TRACK_VISIBLE", "REACQUIRE_TRANSITION"} and followbench_state.last_seen_target_pose_world is not None:
            if args.position == "back":
                task_target = build_back_task_target(robot_pose, followbench_state.last_seen_target_pose_world, real_target_history, target_radius, args.distance)
            else:
                task_target = build_side_task_target(
                    followbench_state.last_seen_target_pose_world,
                    real_target_history,
                    args.position,
                    args.distance,
                    args.protected_person_safe_radius,
                )

            robot_xy = np.asarray(robot_pose, dtype=float).reshape(-1)[:2]
            target_xy = np.asarray(followbench_state.last_seen_target_pose_world, dtype=float).reshape(-1)[:2]
            d_current = float(np.linalg.norm(robot_xy - target_xy))
            target_speed = float(np.asarray(target_person_vel, dtype=float).reshape(-1)[0]) if target_visible else 0.0

            if args.position == "back":
                hold_error = d_current - float(args.distance)
                should_hold = (
                    target_visible
                    and target_speed < args.target_stop_speed_thr
                    and d_current <= float(args.distance) + args.hold_distance_eps
                )
            else:
                desired_pose = task_target.desired_follow_pose_world
                desired_error = float(
                    np.linalg.norm(robot_xy - np.asarray(desired_pose, dtype=float).reshape(-1)[:2])
                ) if desired_pose is not None else float("inf")
                hold_error = desired_error
                should_hold = (
                    target_visible
                    and target_speed < args.target_stop_speed_thr
                    and desired_error <= args.hold_distance_eps
                )
        elif mode == "SEARCH":
            search_xy = compute_search_reference_point(
                robot_pose,
                followbench_state,
                traj_predictor,
                search_history,
                args.search_step,
            )
            task_target = build_search_task_target(robot_pose, followbench_state, search_xy)
        else:
            if followbench_state.last_seen_target_pose_world is not None:
                fallback_traj = np.asarray(followbench_state.last_seen_target_pose_world, dtype=float).reshape(1, 3)[:, :2]
                info = make_idle_info(args.distance, fallback_traj)

        goal_traj = build_goal_traj_list_from_task_target(task_target)
        if enable_step_visuals and goal_traj:
            env.draw_trajectory(goal_traj, traj_type="k-", refresh=True)
            env.draw_points(goal_traj[0], s=60, c="r", refresh=True)

        planner_debug = bool(args.show_planner_debug)
        if mode in {"TRACK_VISIBLE", "REACQUIRE_TRANSITION"} and followbench_state.last_seen_target_pose_world is not None:
            if should_hold:
                hold_history = task_target.planning_target_traj_world_xy[-args.hold_history_points :]
                info = make_idle_info(args.distance, hold_history)
                info["d_current"] = hold_error
                info["d_desired"] = float(args.distance)
            else:
                start_time = time.time()
                opt_vel, info = bso_hfc_planner.control(
                    robot_pose,
                    robot_vel,
                    task_target.planning_target_pose_world,
                    task_target.planning_target_traj_world_xy,
                    env.robot.lidar.local_map,
                    clear_goal_pose_world=task_target.clear_goal_pose_world,
                    clear_goal_radius=task_target.clear_goal_radius,
                    occupied_discs_world=task_target.occupied_discs_world,
                    lidar_range_max=float(env.robot.lidar.range_max),
                    include_edt_debug=planner_debug,
                )
                alg_cost_t = time.time() - start_time
        elif mode == "SEARCH":
            start_time = time.time()
            opt_vel, info = bso_hfc_planner.control(
                robot_pose,
                robot_vel,
                task_target.planning_target_pose_world,
                task_target.planning_target_traj_world_xy,
                env.robot.lidar.local_map,
                clear_goal_pose_world=task_target.clear_goal_pose_world,
                clear_goal_radius=task_target.clear_goal_radius,
                occupied_discs_world=task_target.occupied_discs_world,
                lidar_range_max=float(env.robot.lidar.range_max),
                include_edt_debug=planner_debug,
            )
            alg_cost_t = time.time() - start_time

        if enable_step_visuals:
            if planner_debug:
                draw_planner_debug_overlay(env, info, task_target, real_target_pose_for_display)
            else:
                mpc = np.asarray(info.get("mpc_path_list", np.empty((2, 0), dtype=float)), dtype=float)
                if mpc.size > 0:
                    env.draw_trajectory(mpc, "r", refresh=True)
        pbar.set_postfix_str(timing_monitor.format(info))

        if args.visualize and not args.headless:
            env.save_figure(save_name="step_{:04d}.png".format(i))

        env.step(opt_vel)
        if enable_step_visuals:
            env.render(show_traj=True, show_trail=True)

        if eval_dir != "":
            env.record(alg_cost_t)

        if env.done():
            break

    if eval_dir != "":
        print("evaluating...")
        env.eval(max_steps=max_steps, max_search_steps=max_search_steps)

    if not args.headless:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            env.end(ani_name=f"bso_hfc_demo_{timestamp}", suffix=".gif")
        except (FileNotFoundError, ValueError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=None, help="directory containing scenario yaml files; required unless --world_file is set")
    parser.add_argument("-s", "--scenario", type=str, default="square", help="scenario name")
    parser.add_argument("-p", "--position", type=str, default="back", help="following mode: back, left_side, or right_side")
    parser.add_argument("-d", "--distance", type=float, default=1.3, help="desired following distance d_desired")
    parser.add_argument("-m", "--min_steps", type=int, default=440, help="minimum rollout steps for evaluation")
    parser.add_argument("-i", "--index", type=int, default=0, help="world index used by benchmark batch configs")
    parser.add_argument("-l", "--log_path", type=str, default="", help="evaluation log directory")
    parser.add_argument("-v", "--visualize", action="store_true", default=False, help="whether to render and save figures")
    parser.add_argument("--headless", action="store_true", default=False, help="disable GUI display and per-step visualization for faster evaluation")
    parser.add_argument("--world_file", type=str, default="", help="direct path to a world yaml; overrides config_path/scenario/position")
    parser.add_argument("-t", "--traj_predictor", type=str, default="cvkf", choices=sorted(traj_predictors.keys()), help="trajectory predictor used to generate search reference points (cv, cvkf, sgan)")
    parser.add_argument("--target_stop_speed_thr", type=float, default=0.05, help="speed threshold below which the target is treated as stopped")
    parser.add_argument("--hold_distance_eps", type=float, default=0.15, help="distance tolerance for entering hold mode around the desired pose")
    parser.add_argument("--hold_history_points", type=int, default=32, help="number of recent target-trajectory points kept for hold visualization")
    parser.add_argument("--target_history_maxlen", type=int, default=256, help="maximum number of target history states stored for guidance construction")
    parser.add_argument("--lost_confirm_steps", type=int, default=3, help="frames to wait before switching from tracking to search")
    parser.add_argument("--reacquire_confirm_steps", type=int, default=2, help="frames to confirm target recovery before resuming tracking")
    parser.add_argument("--protected_person_safe_radius", type=float, default=0.1, help="safety radius reserved around the real target during side following")
    parser.add_argument("--search_step", type=float, default=1.0, help="fallback search step when predictor output is unavailable")
    parser.add_argument("--timing_avg_window", type=int, default=10, help="window size used for average timing statistics in the progress bar")
    parser.add_argument(
        "--show_planner_debug",
        action="store_true",
        default=False,
        help="enable EDT sampling in the planner and draw full local-map / hybrid-A* / spline debug layers (default off; matches simple DWA-style viz)",
    )
    args = parser.parse_args()

    if args.world_file != "":
        env_path_file = args.world_file
    elif args.log_path == "":
        env_path_file = args.config_path + "/" + args.scenario + "_" + args.position + ".yaml"
    else:
        env_path_file = os.path.join(args.config_path, args.scenario + "_" + args.position + "_" + str(args.index) + ".yaml")

    main(env_path_file, args)
