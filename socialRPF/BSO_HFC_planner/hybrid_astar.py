from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .local_map import LidarLocalMapAdapter, LocalMapBundle


@dataclass
class HybridAStarConfig:
    """Hybrid A* parameters in yaml order, with robot radius injected last."""

    step_size: float
    angle_res_deg: float
    steering_angles: tuple[float, ...]
    goal_tolerance: float
    max_iterations: int
    robot_radius: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class HybridAStarPlanner:
    """A lightweight local Hybrid A* variant for the robot-centric lidar grid."""

    def __init__(self, cfg: HybridAStarConfig) -> None:
        self.cfg = cfg
        self.angle_res = math.radians(cfg.angle_res_deg)
        self.num_angles = max(8, int(math.ceil(2.0 * math.pi / max(self.angle_res, 1e-6))))

    def plan(
        self,
        start_pose_local: np.ndarray,
        goal_xy_local: np.ndarray,
        map_bundle: LocalMapBundle,
    ) -> np.ndarray | None:
        """Search a coarse kinematically-feasible local path from the robot to the target point."""
        start = np.asarray(start_pose_local, dtype=float).reshape(-1)
        goal = np.asarray(goal_xy_local, dtype=float).reshape(-1)[:2]

        if not self._is_valid(start[0], start[1], map_bundle):
            return None
        if not self._is_valid(goal[0], goal[1], map_bundle):
            return None

        direct_path = self._direct_path(start, goal, map_bundle)
        if direct_path is not None:
            return direct_path

        visited = np.zeros((map_bundle.height, map_bundle.width, self.num_angles), dtype=bool)
        open_list: list[tuple[float, float, tuple[float, float, float], tuple[float, float, float] | None]] = []
        came_from: dict[tuple[int, int, int], tuple[tuple[float, float, float] | None, tuple[float, float, float]]] = {}
        cost_so_far: dict[tuple[int, int, int], float] = {}

        start_state = (float(start[0]), float(start[1]), float(start[2]))
        start_key = self._node_key(start_state[0], start_state[1], start_state[2], map_bundle)
        cost_so_far[start_key] = 0.0
        heapq.heappush(open_list, (self._heuristic(start_state[0], start_state[1], goal), 0.0, start_state, None))

        iterations = 0
        while open_list and iterations < self.cfg.max_iterations:
            iterations += 1
            _, cost, state, parent = heapq.heappop(open_list)
            row, col, theta_idx = self._node_key(state[0], state[1], state[2], map_bundle)

            if not (0 <= row < map_bundle.height and 0 <= col < map_bundle.width):
                continue
            if visited[row, col, theta_idx]:
                continue

            visited[row, col, theta_idx] = True
            came_from[(row, col, theta_idx)] = (parent, state)

            if self._heuristic(state[0], state[1], goal) <= self.cfg.goal_tolerance:
                return self._reconstruct_path(came_from, (row, col, theta_idx), map_bundle)

            if self._segment_is_valid(state[0], state[1], goal[0], goal[1], map_bundle):
                path = self._reconstruct_path(came_from, (row, col, theta_idx), map_bundle)
                goal_heading = float(math.atan2(goal[1] - state[1], goal[0] - state[0]))
                goal_state = np.array([[goal[0], goal[1], goal_heading]], dtype=float)
                if len(path) == 0:
                    return goal_state
                return np.vstack((path, goal_state))

            for delta_heading in self.cfg.steering_angles:
                theta_next = wrap_angle(state[2] + delta_heading)
                x_next = state[0] + self.cfg.step_size * math.cos(theta_next)
                y_next = state[1] + self.cfg.step_size * math.sin(theta_next)
                if not self._segment_is_valid(state[0], state[1], x_next, y_next, map_bundle):
                    continue

                next_key = self._node_key(x_next, y_next, theta_next, map_bundle)
                new_cost = cost + self.cfg.step_size
                if next_key not in cost_so_far or new_cost < cost_so_far[next_key]:
                    cost_so_far[next_key] = new_cost
                    priority = new_cost + self._heuristic(x_next, y_next, goal)
                    heapq.heappush(open_list, (priority, new_cost, (x_next, y_next, theta_next), state))

        return None

    def _direct_path(self, start_pose_local: np.ndarray, goal_xy_local: np.ndarray, map_bundle: LocalMapBundle) -> np.ndarray | None:
        start = np.asarray(start_pose_local, dtype=float).reshape(-1)
        goal = np.asarray(goal_xy_local, dtype=float).reshape(-1)[:2]
        if not self._segment_is_valid(start[0], start[1], goal[0], goal[1], map_bundle):
            return None

        goal_heading = float(math.atan2(goal[1] - start[1], goal[0] - start[0]))
        return np.array(
            [
                [float(start[0]), float(start[1]), float(start[2])],
                [float(goal[0]), float(goal[1]), goal_heading],
            ],
            dtype=float,
        )

    def _theta_to_index(self, theta: float) -> int:
        theta_norm = (theta + 2.0 * math.pi) % (2.0 * math.pi)
        return int(theta_norm / max(self.angle_res, 1e-6)) % self.num_angles

    def _node_key(self, x: float, y: float, theta: float, map_bundle: LocalMapBundle) -> tuple[int, int, int]:
        row, col = LidarLocalMapAdapter.local_to_grid(map_bundle, x, y)
        return row, col, self._theta_to_index(theta)

    def _heuristic(self, x: float, y: float, goal: np.ndarray) -> float:
        return math.hypot(float(goal[0]) - x, float(goal[1]) - y)

    def _is_valid(self, x: float, y: float, map_bundle: LocalMapBundle) -> bool:
        """Use the EDT as a constant-time circular-footprint collision test for a single node."""
        if not LidarLocalMapAdapter.in_bounds(map_bundle, x, y):
            return False
        clearance = LidarLocalMapAdapter.sample_distance_bilinear(map_bundle, x, y)
        return float(clearance) >= float(self.cfg.robot_radius)

    def _segment_is_valid(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        map_bundle: LocalMapBundle,
    ) -> bool:
        """Check a short motion primitive by sampling a few points along the segment."""
        if not self._is_valid(x1, y1, map_bundle):
            return False

        segment_length = math.hypot(x1 - x0, y1 - y0)
        sample_spacing = max(map_bundle.resolution, 1e-3)
        num_checks = max(2, int(math.ceil(segment_length / sample_spacing)))
        alphas = np.linspace(0.0, 1.0, num_checks + 1, dtype=float)[1:]
        sample_points = np.column_stack((
            x0 + alphas * (x1 - x0),
            y0 + alphas * (y1 - y0),
        ))
        clearances = LidarLocalMapAdapter.sample_distances(map_bundle, sample_points)
        return bool(np.all(clearances >= float(self.cfg.robot_radius)))

    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int, int], tuple[tuple[float, float, float] | None, tuple[float, float, float]]],
        terminal_key: tuple[int, int, int],
        map_bundle: LocalMapBundle,
    ) -> np.ndarray:
        path_states: list[tuple[float, float, float]] = []
        current_key = terminal_key
        while current_key in came_from:
            parent_state, state = came_from[current_key]
            path_states.append(state)
            if parent_state is None:
                break
            current_key = self._node_key(parent_state[0], parent_state[1], parent_state[2], map_bundle)
        path_states.reverse()
        return np.asarray(path_states, dtype=float)
