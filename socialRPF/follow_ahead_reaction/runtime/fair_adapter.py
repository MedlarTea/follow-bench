from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .base import inflate_occupancy


def compute_baseline_eval_horizons(env, min_steps: int) -> Tuple[int, int]:
    """Match the baseline scripts' evaluation semantics.

    Baselines interpret `-m` as a minimum trajectory length:
    - static / target-only scenes: max_steps = min_steps
    - scenes with extra humans:    max_steps = min_steps * 2
    - all cases: max_search_steps = int(min_steps * 0.5)
    """
    if len(getattr(env, "human_list", [])) == 0:
        max_steps = int(min_steps)
    else:
        max_steps = int(min_steps) * 2
    max_search_steps = int(min_steps * 0.5)
    return max_steps, max_search_steps


def get_visible_target(env):
    for obj in getattr(env, "visible_object_list", []):
        if getattr(obj, "role", "") == "target":
            return obj
    return None


def get_visible_humans(env):
    humans = []
    for obj in getattr(env, "visible_object_list", []):
        if getattr(obj, "role", "") == "human":
            humans.append(obj)
    return humans


def scan_points_global(robot_state, scan_data):
    ranges = np.asarray(scan_data["ranges"], dtype=float)
    angles = np.linspace(scan_data["angle_min"], scan_data["angle_max"], len(ranges))

    state = np.asarray(robot_state, dtype=float).reshape(-1)
    if state.size < 3:
        raise ValueError("robot_state must contain at least [x, y, yaw]")

    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )

    range_min = float(scan_data.get("range_min", 0.0))
    range_max = float(scan_data.get("range_max", np.inf))

    points = []
    for rng, ang in zip(ranges, angles):
        if not np.isfinite(rng):
            continue
        if rng <= range_min or rng >= (range_max - 0.01):
            continue
        local = np.array([rng * math.cos(ang), rng * math.sin(ang)], dtype=float)
        global_point = np.array([x, y], dtype=float) + rot @ local
        points.append(global_point)

    if not points:
        return np.empty((0, 2), dtype=float)
    return np.vstack(points)


def _rasterize_disc(grid_map, origin_x, origin_y, resolution, center_xy, radius):
    cx = int(np.rint((center_xy[0] - origin_x) / resolution))
    cy = int(np.rint((center_xy[1] - origin_y) / resolution))
    radius_cells = max(1, int(np.ceil(radius / resolution)))

    height, width = grid_map.shape
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            gx = cx + dx
            gy = cy + dy
            if 0 <= gx < width and 0 <= gy < height:
                grid_map[gy, gx] = 100


def build_perception_costmap_from_env(
    env,
    robot_state,
    *,
    resolution: float = 0.1,
    inflation_radius: float = 0.55,
    human_padding: float = 0.15,
    min_half_extent: float = 6.0,
) -> Tuple[Dict[str, object], np.ndarray]:
    """Build a robot-centric occupancy map from lidar + visible humans only.

    This intentionally avoids the global obstacle list used by the original
    Follow-Reaction benchmark adapter.
    """
    scan_data = env.get_lidar_scan()
    range_max = float(scan_data.get("range_max", min_half_extent))
    half_extent = max(min_half_extent, range_max + 0.5)

    robot_state = np.asarray(robot_state, dtype=float).reshape(-1)
    robot_xy = robot_state[:2]
    origin_x = float(robot_xy[0] - half_extent)
    origin_y = float(robot_xy[1] - half_extent)
    width = max(1, int(np.ceil((2.0 * half_extent) / resolution)))
    height = max(1, int(np.ceil((2.0 * half_extent) / resolution)))

    grid_map = np.zeros((height, width), dtype=np.uint8)

    world = getattr(env, "_world", None)
    if world is not None:
        x_min, x_max = map(float, getattr(world, "x_range", [origin_x, origin_x + 2.0 * half_extent]))
        y_min, y_max = map(float, getattr(world, "y_range", [origin_y, origin_y + 2.0 * half_extent]))

        x_coords = origin_x + np.arange(width, dtype=float) * resolution
        y_coords = origin_y + np.arange(height, dtype=float) * resolution

        invalid_x = (x_coords < x_min) | (x_coords > x_max)
        invalid_y = (y_coords < y_min) | (y_coords > y_max)
        if np.any(invalid_x):
            grid_map[:, invalid_x] = 100
        if np.any(invalid_y):
            grid_map[invalid_y, :] = 100

    visible_target = get_visible_target(env)
    visible_target_xy: Optional[np.ndarray] = None
    visible_target_radius = 0.0
    if visible_target is not None:
        visible_target_xy = np.asarray(visible_target.state[:2, 0], dtype=float)
        visible_target_radius = float(getattr(visible_target, "radius", 0.0))

    scan_points = scan_points_global(robot_state, scan_data)
    if scan_points.size > 0 and visible_target_xy is not None:
        dists = np.linalg.norm(scan_points - visible_target_xy[None, :], axis=1)
        scan_points = scan_points[dists > (visible_target_radius + 0.2)]

    point_radius = max(resolution * 1.5, 0.12)
    for point in scan_points:
        _rasterize_disc(grid_map, origin_x, origin_y, resolution, point, point_radius)

    visible_humans = get_visible_humans(env)
    for human in visible_humans:
        center = np.asarray(human.state[:2, 0], dtype=float)
        radius = float(getattr(human, "radius", 0.3)) + human_padding
        _rasterize_disc(grid_map, origin_x, origin_y, resolution, center, radius)

    inflation_cells = int(np.ceil(inflation_radius / resolution))
    inflated_map = inflate_occupancy(grid_map, inflation_cells)
    costmap = np.where(inflated_map > 0, 100, 0).astype(np.int8)

    params = {
        "map_origin_x": origin_x,
        "map_origin_y": origin_y,
        "map_res": resolution,
        "map_width": int(costmap.shape[1]),
        "map_data": costmap.flatten().tolist(),
        "inflation_radius": float(inflation_radius),
        "inflation_cells": int(inflation_cells),
        "perception_map_half_extent": float(half_extent),
        "perception_visible_humans": len(visible_humans),
        "perception_scan_points": int(scan_points.shape[0]),
    }
    return params, costmap
