from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import yaml

from .bspline_opt import BSplineConfig, BSplineOptimizer, BSplineResult
from .hybrid_astar import HybridAStarConfig, HybridAStarPlanner
from .local_map import LidarLocalMapAdapter
from .mpc_tracker import MPCConfig, MPCTracker

DEFAULT_PARAM_YAML_PATH = Path(__file__).with_name("bso_hfc_params.yaml")


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class AdaptiveTimingConfig:
    v_ref: float
    v_ref_min: float
    v_ref_max: float
    k_p: float
    k_i: float
    k_d: float
    d_j_min: float


@dataclass
class DiscSpec:
    center_world_xy: np.ndarray
    radius: float


@dataclass
class BSOHFCConfig:
    """Merged runtime config for the extracted BSO-HFC pipeline."""

    target_clear_margin: float
    target_radius: float
    occ_threshold: float
    unknown_is_occupied: bool
    local_map_window_size_m: float | None
    d_desired: float
    adaptive_timing: AdaptiveTimingConfig
    hybrid_astar: HybridAStarConfig
    bspline: BSplineConfig
    mpc: MPCConfig


class IncrementalPID:
    """Incremental PID used for the paper's adaptive v_ref update."""

    def __init__(self, k_p: float, k_i: float, k_d: float) -> None:
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.reset()

    def reset(self) -> None:
        self.prev_error = 0.0
        self.prev_prev_error = 0.0
        self.output = 0.0

    def update(self, error: float, dt: float) -> float:
        dt = max(float(dt), 1e-6)
        delta = (
            self.k_p * (error - self.prev_error)
            + self.k_i * error * dt
            + self.k_d * (error - 2.0 * self.prev_error + self.prev_prev_error) / dt
        )
        self.output += delta
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        return float(self.output)


class GuidanceBuilder:
    """Build the paper's guidance path from the target trajectory only."""

    def __init__(self, map_adapter: LidarLocalMapAdapter) -> None:
        self.map_adapter = map_adapter

    def build(
        self,
        target_traj_world_xy: np.ndarray | list,
        robot_pose_world: np.ndarray,
        target_pose_world: np.ndarray,
        map_bundle,
        prev_anchor_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        traj_world = self._coerce_points_xy(target_traj_world_xy)
        target_xy_world = np.asarray(target_pose_world, dtype=float).reshape(-1)[:2].reshape(1, 2)
        robot_xy_world = np.asarray(robot_pose_world, dtype=float).reshape(-1)[:2]

        if traj_world.size == 0:
            path_world = target_xy_world
        else:
            path_world = traj_world.copy()
            if np.linalg.norm(path_world[-1] - target_xy_world[0]) > 1e-6:
                path_world = np.vstack((path_world, target_xy_world))
            else:
                path_world[-1] = target_xy_world[0]

        path_world, anchor_world = self._trim_to_robot_relevant_suffix(path_world, robot_xy_world, prev_anchor_world)
        path_local = self.map_adapter.world_to_local(robot_pose_world, path_world)
        x_min, x_max, y_min, y_max = self.map_adapter.map_limits(map_bundle)
        inside = (
            (path_local[:, 0] >= x_min)
            & (path_local[:, 0] <= x_max)
            & (path_local[:, 1] >= y_min)
            & (path_local[:, 1] <= y_max)
        )
        path_local = path_local[inside]

        target_local = self.map_adapter.world_to_local(robot_pose_world, target_xy_world)[0]
        target_local[0] = clamp(target_local[0], x_min, x_max)
        target_local[1] = clamp(target_local[1], y_min, y_max)
        if path_local.size == 0:
            path_local = target_local.reshape(1, 2)
        elif np.linalg.norm(path_local[-1] - target_local) > 0.5 * map_bundle.resolution:
            path_local = np.vstack((path_local, target_local))
        else:
            path_local[-1] = target_local

        return self._remove_duplicate_points(path_local, epsilon=0.5 * map_bundle.resolution), np.asarray(anchor_world, dtype=float)

    def _trim_to_robot_relevant_suffix(
        self,
        path_world: np.ndarray,
        robot_xy_world: np.ndarray,
        prev_anchor_world: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(path_world, dtype=float)
        if len(points) <= 1:
            return points, points.reshape(-1, 2)[0]

        min_segment_idx = 0
        if prev_anchor_world is not None:
            prev_idx, _ = self._closest_projection_on_segments(points, np.asarray(prev_anchor_world, dtype=float).reshape(-1)[:2], 0)
            min_segment_idx = int(prev_idx)

        anchor_idx, anchor_point = self._closest_projection_on_segments(points, robot_xy_world, min_segment_idx)
        suffix_tail = points[anchor_idx + 1 :]
        if len(suffix_tail) > 0:
            suffix = np.vstack((anchor_point.reshape(1, 2), suffix_tail))
        else:
            suffix = anchor_point.reshape(1, 2)
        return self._remove_duplicate_points(suffix), anchor_point

    def _closest_projection_on_segments(
        self,
        points: np.ndarray,
        query_xy_world: np.ndarray,
        min_segment_idx: int,
    ) -> tuple[int, np.ndarray]:
        query = np.asarray(query_xy_world, dtype=float).reshape(-1)[:2]
        segment_start = max(int(min_segment_idx), 0)
        segment_end = max(len(points) - 1, 1)

        best_idx = segment_start
        best_projection = np.asarray(points[segment_start], dtype=float)
        best_dist_sq = float("inf")
        for idx in range(segment_start, segment_end):
            start = np.asarray(points[idx], dtype=float)
            end = np.asarray(points[idx + 1], dtype=float)
            seg = end - start
            seg_norm_sq = float(np.dot(seg, seg))
            if seg_norm_sq <= 1e-12:
                projection = start
            else:
                alpha = float(np.dot(query - start, seg) / seg_norm_sq)
                alpha = clamp(alpha, 0.0, 1.0)
                projection = start + alpha * seg

            dist_sq = float(np.sum((projection - query) ** 2))
            if dist_sq < best_dist_sq:
                best_idx = idx
                best_projection = projection
                best_dist_sq = dist_sq

        return best_idx, np.asarray(best_projection, dtype=float)

    @staticmethod
    def _coerce_points_xy(points: np.ndarray | list) -> np.ndarray:
        if isinstance(points, list):
            if len(points) == 0:
                return np.empty((0, 2), dtype=float)
            rows = []
            for point in points:
                arr = np.asarray(point, dtype=float).reshape(-1)
                if arr.size >= 2:
                    rows.append(arr[:2])
            return np.asarray(rows, dtype=float) if rows else np.empty((0, 2), dtype=float)

        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)[:, :2]
        if arr.shape[0] in (2, 3) and arr.shape[1] > arr.shape[0]:
            return arr[:2, :].T
        return arr[:, :2]

    @staticmethod
    def _remove_duplicate_points(points_local: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        points = np.asarray(points_local, dtype=float)
        if len(points) <= 1:
            return points

        unique = [points[0]]
        threshold = max(float(epsilon), 1e-6)
        for point in points[1:]:
            if np.linalg.norm(point - unique[-1]) > threshold:
                unique.append(point)
        return np.asarray(unique, dtype=float)


def _read_param_yaml(config_path: str | Path = DEFAULT_PARAM_YAML_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"BSO-HFC parameter yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"BSO-HFC parameter yaml is empty: {path}")
    if not isinstance(data, Mapping):
        raise ValueError("BSO-HFC parameter yaml must contain a mapping at the top level.")
    return dict(data)


def _build_config(
    cfg_dict: dict[str, Any],
    *,
    sample_time: float,
    linear_min: float,
    linear_max: float,
    omega_min: float,
    omega_max: float,
    acc_v: float,
    acc_omega: float,
    robot_radius: float,
    target_radius: float,
    d_desired: float,
) -> BSOHFCConfig:
    target_clear_margin = max(float(cfg_dict.get("target_clear_margin", 0.15)), 0.0)
    occ_threshold = clamp(float(cfg_dict.get("occ_threshold", 0.65)), 0.0, 1.0)
    unknown_is_occupied = bool(cfg_dict.get("unknown_is_occupied", False))
    raw_window_size = cfg_dict.get("local_map_window_size_m", None)
    local_map_window_size_m = None if raw_window_size in (None, "", "null") else max(float(raw_window_size), 1e-3)
    d_desired = max(float(d_desired), 1e-3)

    adaptive = dict(cfg_dict["adaptive_timing"])
    v_ref_min = clamp(float(adaptive["v_ref_min"]), max(0.0, linear_min), linear_max)
    v_ref_max = clamp(float(adaptive["v_ref_max"]), v_ref_min, linear_max)
    adaptive["v_ref"] = clamp(float(adaptive["v_ref"]), v_ref_min, v_ref_max)
    adaptive["v_ref_min"] = v_ref_min
    adaptive["v_ref_max"] = v_ref_max
    adaptive["k_p"] = float(adaptive["k_p"])
    adaptive["k_i"] = float(adaptive["k_i"])
    adaptive["k_d"] = float(adaptive["k_d"])
    adaptive["d_j_min"] = max(float(adaptive["d_j_min"]), 1e-4)

    hybrid = dict(cfg_dict["hybrid_astar"])
    hybrid["step_size"] = max(float(hybrid["step_size"]), 1e-3)
    hybrid["angle_res_deg"] = max(float(hybrid["angle_res_deg"]), 1e-3)
    hybrid["steering_angles"] = tuple(float(v) for v in hybrid["steering_angles"])
    hybrid["goal_tolerance"] = max(float(hybrid["goal_tolerance"]), 1e-3)
    hybrid["max_iterations"] = max(int(hybrid["max_iterations"]), 1)
    hybrid["robot_radius"] = float(robot_radius)

    bspline = dict(cfg_dict["bspline"])
    bspline["p"] = max(int(bspline["p"]), 1)
    bspline["num_ctrl_points"] = max(int(bspline["num_ctrl_points"]), bspline["p"] + 2)
    bspline["num_samples"] = max(int(bspline["num_samples"]), 2)
    bspline["optimize_maxiter"] = max(int(bspline["optimize_maxiter"]), 1)
    bspline["d_thr"] = max(float(bspline["d_thr"]), 0.0)
    bspline["omega_c"] = max(float(bspline["omega_c"]), 0.0)
    bspline["omega_g"] = max(float(bspline["omega_g"]), 0.0)
    bspline["omega_s"] = max(float(bspline["omega_s"]), 0.0)
    bspline["omega_d"] = max(float(bspline["omega_d"]), 0.0)
    bspline["v_max"] = clamp(float(bspline["v_max"]), 1e-3, max(linear_max, 1e-3))
    bspline["a_max"] = clamp(float(bspline["a_max"]), 1e-3, max(acc_v, 1e-3))

    mpc = dict(cfg_dict["mpc"])
    mpc["horizon"] = max(int(mpc["horizon"]), 2)
    mpc["max_iter"] = max(int(mpc["max_iter"]), 1)
    mpc["warm_start"] = bool(mpc["warm_start"])
    mpc["allow_reverse"] = bool(mpc.get("allow_reverse", False))
    for key in (
        "q_pos",
        "q_yaw",
        "q_v",
        "q_omega",
        "qf_pos",
        "qf_yaw",
        "qf_v",
        "qf_omega",
        "r_acc_v",
        "r_acc_omega",
        "rd_acc_v",
        "rd_acc_omega",
    ):
        mpc[key] = max(float(mpc[key]), 0.0)
    mpc["dt"] = float(sample_time)
    mpc["min_v"] = linear_min if mpc["allow_reverse"] else max(0.0, linear_min)
    mpc["max_v"] = linear_max
    mpc["min_omega"] = omega_min
    mpc["max_omega"] = omega_max
    mpc["max_acc_v"] = max(0.2, acc_v)
    mpc["max_acc_omega"] = max(0.2, acc_omega)

    return BSOHFCConfig(
        target_clear_margin=target_clear_margin,
        target_radius=max(float(target_radius), 0.0),
        occ_threshold=occ_threshold,
        unknown_is_occupied=unknown_is_occupied,
        local_map_window_size_m=local_map_window_size_m,
        d_desired=d_desired,
        adaptive_timing=AdaptiveTimingConfig(**adaptive),
        hybrid_astar=HybridAStarConfig(**hybrid),
        bspline=BSplineConfig(**bspline),
        mpc=MPCConfig(**mpc),
    )


def load_bso_hfc_config(
    robot_tuple,
    sample_time: float,
    d_desired: float,
    target_radius: float = 0.0,
    config_path: str | Path = DEFAULT_PARAM_YAML_PATH,
) -> BSOHFCConfig:
    """Public config loader for the follow-bench BSO-HFC module."""
    max_speed = np.asarray(robot_tuple.max_speed, dtype=float).reshape(-1)
    min_speed = np.asarray(robot_tuple.min_speed, dtype=float).reshape(-1)
    max_acce = np.asarray(robot_tuple.max_acce, dtype=float).reshape(-1)
    robot_radius = float(getattr(robot_tuple, "radius", 0.35))

    linear_min = float(min_speed[0]) if len(min_speed) > 0 else 0.0
    linear_max = float(max_speed[0]) if len(max_speed) > 0 else 1.0
    omega_min = float(min_speed[1]) if len(min_speed) > 1 else -1.0
    omega_max = float(max_speed[1]) if len(max_speed) > 1 else 1.0
    acc_v = float(max_acce[0]) if len(max_acce) > 0 else 0.5
    acc_omega = float(max_acce[1]) if len(max_acce) > 1 else 1.0

    cfg_dict = _read_param_yaml(config_path)
    return _build_config(
        cfg_dict,
        sample_time=sample_time,
        linear_min=linear_min,
        linear_max=linear_max,
        omega_min=omega_min,
        omega_max=omega_max,
        acc_v=max(0.2, acc_v),
        acc_omega=max(0.2, acc_omega),
        robot_radius=robot_radius,
        target_radius=max(float(target_radius), 0.0),
        d_desired=float(d_desired),
    )


class BSOHFCPlanner:
    """Pure BSO-HFC local planner for the follow-bench integration."""

    def __init__(self, cfg: BSOHFCConfig) -> None:
        self.cfg = cfg
        self.map_adapter = LidarLocalMapAdapter(
            occ_threshold=cfg.occ_threshold,
            clear_target_margin=cfg.target_clear_margin,
            unknown_is_occupied=cfg.unknown_is_occupied,
        )
        self.guidance_builder = GuidanceBuilder(self.map_adapter)
        self.hybrid_astar = HybridAStarPlanner(cfg.hybrid_astar)
        self.bspline = BSplineOptimizer(cfg.bspline)
        self.tracker = MPCTracker(cfg.mpc)
        self.timing_pid = IncrementalPID(
            cfg.adaptive_timing.k_p,
            cfg.adaptive_timing.k_i,
            cfg.adaptive_timing.k_d,
        )
        self.prev_guidance_anchor_world: np.ndarray | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.timing_pid.reset()
        self.prev_guidance_anchor_world = None

    def control(
        self,
        robot_pose_world: np.ndarray,
        robot_vel: np.ndarray,
        target_pose_world: np.ndarray | None,
        target_traj_world_xy: np.ndarray | list,
        local_map,
        clear_goal_pose_world: np.ndarray | None = None,
        clear_goal_radius: float = 0.0,
        occupied_discs_world: list[DiscSpec] | None = None,
        lidar_range_max: float | None = None,
        include_edt_debug: bool = False,
    ) -> tuple[np.ndarray, dict]:
        pose_world = np.asarray(robot_pose_world, dtype=float).reshape(-1)
        vel = np.asarray(robot_vel, dtype=float).reshape(-1)
        target_world = self._coerce_target_pose(target_pose_world, pose_world)
        clear_goal_world = self._coerce_target_pose(clear_goal_pose_world, pose_world)
        occupied_discs = self._coerce_discs(occupied_discs_world)

        if local_map is None or target_world is None:
            return np.zeros((2, 1), dtype=float), self._empty_debug_info(False)

        if clear_goal_world is None and not occupied_discs:
            clear_goal_world = target_world.copy()
            clear_goal_radius = float(self.cfg.target_radius)

        stage_timing_ms = {
            "hybrid_astar": 0.0,
            "hybrid_astar_search": 0.0,
            "fallback": 0.0,
            "bspline": 0.0,
            "mpc": 0.0,
        }

        observable_radius = None if lidar_range_max is None else max(float(lidar_range_max), 0.0)
        map_bundle = self.map_adapter.build(
            local_map,
            pose_world,
            clear_goal_pose_world=clear_goal_world,
            clear_goal_radius=clear_goal_radius,
            occupied_discs_world=occupied_discs,
            observable_radius=observable_radius,
            window_size_m=self.cfg.local_map_window_size_m,
        )
        target_raw_local = self.map_adapter.world_to_local(pose_world, target_world[None, :2])[0]
        local_goal_raw_local, target_outside_local_map = self._compute_local_goal(target_raw_local, map_bundle)
        local_goal_clamped_local = self._clamp_local_point(local_goal_raw_local, map_bundle)
        local_goal_projected_local = self._project_goal_to_free_space(local_goal_clamped_local, map_bundle)
        guidance_path_local, guidance_anchor_world = self.guidance_builder.build(
            target_traj_world_xy,
            pose_world,
            target_world[:2],
            map_bundle,
            prev_anchor_world=self.prev_guidance_anchor_world,
        )
        self.prev_guidance_anchor_world = guidance_anchor_world.copy()

        astar_start = perf_counter()
        astar_path = self.hybrid_astar.plan(
            np.array([0.0, 0.0, 0.0], dtype=float),
            local_goal_projected_local,
            map_bundle,
        )
        search_ms = (perf_counter() - astar_start) * 1000.0
        fallback_ms = 0.0
        fallback_used = False
        if astar_path is None or len(astar_path) < 2:
            fallback_used = True
            fallback_start = perf_counter()
            astar_path = self._build_fallback_path_to_goal(local_goal_projected_local, map_bundle)
            fallback_ms = (perf_counter() - fallback_start) * 1000.0
        stage_timing_ms["hybrid_astar"] = search_ms + fallback_ms
        stage_timing_ms["hybrid_astar_search"] = search_ms
        stage_timing_ms["fallback"] = fallback_ms

        bspline_start = perf_counter()
        seed_control_points = self.bspline.build_seed_control_points(astar_path)
        delta_t, v_ref, d_j = self._compute_timing(pose_world, target_world, seed_control_points)
        spline_result = self.bspline.optimize(
            seed_control_points=seed_control_points,
            guidance_path_local=guidance_path_local,
            map_bundle=map_bundle,
            delta_t=delta_t,
        )
        stage_timing_ms["bspline"] = (perf_counter() - bspline_start) * 1000.0

        mpc_start = perf_counter()
        opt_vel, mpc_path_local = self.tracker.control(vel, spline_result)
        stage_timing_ms["mpc"] = (perf_counter() - mpc_start) * 1000.0

        info = self._build_debug_info(
            planning_success=True,
            astar_local_xy=astar_path[:, :2],
            guidance_path_local=guidance_path_local,
            spline_result=spline_result,
            mpc_local_xy=mpc_path_local,
            robot_pose_world=pose_world,
            stage_timing_ms=stage_timing_ms,
            target_world=target_world,
            clear_goal_world=clear_goal_world,
            occupied_discs_world=occupied_discs,
            target_raw_local=target_raw_local,
            local_goal_raw_local=local_goal_raw_local,
            local_goal_clamped_local=local_goal_clamped_local,
            local_goal_projected_local=local_goal_projected_local,
            target_outside_local_map=target_outside_local_map,
            map_bundle=map_bundle,
            fallback_used=fallback_used,
            v_ref=v_ref,
            delta_t=delta_t,
            d_j=d_j,
            observable_radius=observable_radius,
            include_edt_debug=include_edt_debug,
            requested_local_map_window_size_m=self.cfg.local_map_window_size_m,
        )
        return opt_vel, info

    def _coerce_target_pose(self, target_pose: np.ndarray | None, robot_pose_world: np.ndarray) -> np.ndarray | None:
        if target_pose is None:
            return None
        arr = np.asarray(target_pose, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        if arr.size >= 3:
            return arr[:3]
        yaw = float(np.arctan2(arr[1] - robot_pose_world[1], arr[0] - robot_pose_world[0]))
        return np.array([arr[0], arr[1], yaw], dtype=float)

    def _coerce_discs(self, occupied_discs_world: list[DiscSpec] | None) -> list[DiscSpec]:
        discs = []
        for disc in occupied_discs_world or []:
            if isinstance(disc, DiscSpec):
                center_world_xy = np.asarray(disc.center_world_xy, dtype=float).reshape(-1)[:2]
                discs.append(DiscSpec(center_world_xy=center_world_xy, radius=max(float(disc.radius), 0.0)))
                continue

            if hasattr(disc, "center_world_xy") and hasattr(disc, "radius"):
                center_world_xy = np.asarray(getattr(disc, "center_world_xy"), dtype=float).reshape(-1)[:2]
                radius = max(float(getattr(disc, "radius")), 0.0)
                discs.append(DiscSpec(center_world_xy=center_world_xy, radius=radius))
                continue

            if isinstance(disc, dict):
                center_world_xy = np.asarray(disc.get("center_world_xy", disc.get("center_world")), dtype=float).reshape(-1)[:2]
                radius = max(float(disc.get("radius", 0.0)), 0.0)
                discs.append(DiscSpec(center_world_xy=center_world_xy, radius=radius))
                continue

            if isinstance(disc, (tuple, list)) and len(disc) >= 2:
                center_world_xy = np.asarray(disc[0], dtype=float).reshape(-1)[:2]
                radius = max(float(disc[1]), 0.0)
                discs.append(DiscSpec(center_world_xy=center_world_xy, radius=radius))
                continue

            raise ValueError("occupied_discs_world entries must expose center_world_xy and radius.")

        return discs

    def _clamp_local_point(self, point_local: np.ndarray, map_bundle) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.map_adapter.map_limits(map_bundle)
        point = np.asarray(point_local, dtype=float).reshape(-1)[:2].copy()
        point[0] = clamp(point[0], x_min, x_max)
        point[1] = clamp(point[1], y_min, y_max)
        return point

    def _compute_local_goal(self, target_raw_local: np.ndarray, map_bundle) -> tuple[np.ndarray, bool]:
        target = np.asarray(target_raw_local, dtype=float).reshape(-1)[:2]
        if self._is_inside_local_window(target, map_bundle):
            return target.copy(), False

        boundary_point = self._intersect_ray_with_local_window(target, map_bundle)
        shrunk_point = self._shrink_local_goal_inside_window(boundary_point, target, map_bundle)
        return shrunk_point, True

    def _is_inside_local_window(self, point_local: np.ndarray, map_bundle) -> bool:
        x_min, x_max, y_min, y_max = self.map_adapter.map_limits(map_bundle)
        point = np.asarray(point_local, dtype=float).reshape(-1)[:2]
        return bool(x_min <= point[0] <= x_max and y_min <= point[1] <= y_max)

    def _intersect_ray_with_local_window(self, target_raw_local: np.ndarray, map_bundle) -> np.ndarray:
        target = np.asarray(target_raw_local, dtype=float).reshape(-1)[:2]
        norm = float(np.linalg.norm(target))
        if norm <= 1e-9:
            return np.zeros((2,), dtype=float)

        x_min, x_max, y_min, y_max = self.map_adapter.map_limits(map_bundle)
        candidates = []
        if abs(target[0]) > 1e-9:
            x_bound = x_max if target[0] > 0.0 else x_min
            alpha = x_bound / target[0]
            if alpha > 0.0:
                y = alpha * target[1]
                if y_min - 1e-9 <= y <= y_max + 1e-9:
                    candidates.append((alpha, np.array([x_bound, y], dtype=float)))
        if abs(target[1]) > 1e-9:
            y_bound = y_max if target[1] > 0.0 else y_min
            alpha = y_bound / target[1]
            if alpha > 0.0:
                x = alpha * target[0]
                if x_min - 1e-9 <= x <= x_max + 1e-9:
                    candidates.append((alpha, np.array([x, y_bound], dtype=float)))

        if not candidates:
            return self._clamp_local_point(target, map_bundle)
        _, point = min(candidates, key=lambda item: item[0])
        return point

    def _shrink_local_goal_inside_window(self, boundary_point_local: np.ndarray, target_raw_local: np.ndarray, map_bundle) -> np.ndarray:
        boundary = np.asarray(boundary_point_local, dtype=float).reshape(-1)[:2]
        target = np.asarray(target_raw_local, dtype=float).reshape(-1)[:2]
        norm = float(np.linalg.norm(target))
        if norm <= 1e-9:
            return boundary

        direction = target / norm
        margin = max(float(self.cfg.hybrid_astar.robot_radius), float(map_bundle.resolution))
        return self._clamp_local_point(boundary - margin * direction, map_bundle)

    def _project_goal_to_free_space(self, goal_local: np.ndarray, map_bundle) -> np.ndarray:
        goal = np.asarray(goal_local, dtype=float).reshape(-1)[:2].copy()
        if self._is_local_position_valid(goal, map_bundle):
            return goal

        distance = float(np.linalg.norm(goal))
        if distance <= 1e-9:
            return np.zeros((2,), dtype=float)

        step = max(min(self.cfg.hybrid_astar.step_size, map_bundle.resolution), 1e-3)
        num_checks = max(2, int(np.ceil(distance / step)))
        alphas = np.linspace(1.0, 0.0, num_checks + 1, dtype=float)
        for alpha in alphas[1:]:
            candidate = alpha * goal
            if self._is_local_position_valid(candidate, map_bundle):
                return candidate
        return np.zeros((2,), dtype=float)

    def _compute_timing(
        self,
        robot_pose_world: np.ndarray,
        target_pose_world: np.ndarray,
        control_points_local: np.ndarray,
    ) -> tuple[float, float, float]:
        d_j = self.cfg.adaptive_timing.d_j_min
        if len(control_points_local) > 1:
            d_j = float(np.mean(np.linalg.norm(np.diff(control_points_local, axis=0), axis=1)))
        d_j = max(d_j, self.cfg.adaptive_timing.d_j_min)

        d_current = float(np.linalg.norm(robot_pose_world[:2] - target_pose_world[:2]))
        error = d_current - self.cfg.d_desired
        correction = self.timing_pid.update(error, self.cfg.mpc.dt)
        v_ref = clamp(
            self.cfg.adaptive_timing.v_ref + correction,
            self.cfg.adaptive_timing.v_ref_min,
            self.cfg.adaptive_timing.v_ref_max,
        )
        delta_t = float(d_j / max(v_ref, 1e-3))
        return delta_t, v_ref, d_j

    def _is_local_position_valid(self, point_local: np.ndarray, map_bundle) -> bool:
        point = np.asarray(point_local, dtype=float).reshape(-1)[:2]
        if not self.map_adapter.in_bounds(map_bundle, point[0], point[1]):
            return False
        clearance = self.map_adapter.sample_distance_bilinear(map_bundle, point[0], point[1])
        return float(clearance) >= float(self.cfg.hybrid_astar.robot_radius)

    def _is_local_segment_valid(self, start_local: np.ndarray, end_local: np.ndarray, map_bundle) -> bool:
        start = np.asarray(start_local, dtype=float).reshape(-1)[:2]
        end = np.asarray(end_local, dtype=float).reshape(-1)[:2]
        if not self._is_local_position_valid(end, map_bundle):
            return False

        segment_length = float(np.linalg.norm(end - start))
        sample_spacing = max(map_bundle.resolution, 1e-3)
        num_checks = max(2, int(np.ceil(segment_length / sample_spacing)))
        alphas = np.linspace(0.0, 1.0, num_checks + 1, dtype=float)[1:]
        sample_points = start[None, :] + alphas[:, None] * (end - start)[None, :]
        clearances = self.map_adapter.sample_distances(map_bundle, sample_points)
        return bool(np.all(clearances >= float(self.cfg.hybrid_astar.robot_radius)))

    def _safe_line_prefix(self, start_local: np.ndarray, goal_local: np.ndarray, map_bundle) -> np.ndarray:
        start = np.asarray(start_local, dtype=float).reshape(-1)[:2]
        goal = np.asarray(goal_local, dtype=float).reshape(-1)[:2]
        if not self._is_local_position_valid(start, map_bundle):
            return start[None, :]

        distance = float(np.linalg.norm(goal - start))
        if distance <= 1e-9:
            return start[None, :]

        sample_spacing = max(min(self.cfg.hybrid_astar.step_size, map_bundle.resolution), 1e-3)
        num_steps = max(1, int(np.ceil(distance / sample_spacing)))
        alphas = np.linspace(0.0, 1.0, num_steps + 1, dtype=float)[1:]

        points = [start]
        previous = start
        for alpha in alphas:
            candidate = start + alpha * (goal - start)
            if not self._is_local_segment_valid(previous, candidate, map_bundle):
                break
            points.append(candidate)
            previous = candidate
        return np.asarray(points, dtype=float)

    def _build_fallback_path_to_goal(self, goal_local: np.ndarray, map_bundle) -> np.ndarray:
        origin = np.zeros((2,), dtype=float)
        goal = np.asarray(goal_local, dtype=float).reshape(-1)[:2]

        best_points = self._safe_line_prefix(origin, goal, map_bundle)
        best_error = float(np.linalg.norm(best_points[-1] - goal))
        step_size = max(self.cfg.hybrid_astar.step_size, map_bundle.resolution)

        for delta_heading in self.cfg.hybrid_astar.steering_angles:
            candidate = step_size * np.array([np.cos(delta_heading), np.sin(delta_heading)], dtype=float)
            if not self._is_local_segment_valid(origin, candidate, map_bundle):
                continue

            tail_points = self._safe_line_prefix(candidate, goal, map_bundle)
            candidate_points = np.vstack((origin[None, :], tail_points))
            candidate_points = self._remove_duplicate_points(candidate_points, epsilon=0.5 * map_bundle.resolution)
            candidate_error = float(np.linalg.norm(candidate_points[-1] - goal))
            if candidate_error < best_error - 1e-6 or (
                abs(candidate_error - best_error) <= 1e-6 and len(candidate_points) > len(best_points)
            ):
                best_points = candidate_points
                best_error = candidate_error

        return self._points_to_local_path(best_points)

    @staticmethod
    def _points_to_local_path(points_local: np.ndarray) -> np.ndarray:
        points = np.asarray(points_local, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        headings = np.zeros((len(points),), dtype=float)
        if len(points) > 1:
            diffs = np.diff(points[:, :2], axis=0)
            headings[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
            headings[-1] = headings[-2]
        return np.column_stack((points[:, :2], headings))

    @staticmethod
    def _remove_duplicate_points(points_local: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        points = np.asarray(points_local, dtype=float)
        if len(points) <= 1:
            return points

        unique = [points[0]]
        threshold = max(float(epsilon), 1e-6)
        for point in points[1:]:
            if np.linalg.norm(point - unique[-1]) > threshold:
                unique.append(point)
        return np.asarray(unique, dtype=float)

    def _build_debug_info(
        self,
        planning_success: bool,
        astar_local_xy: np.ndarray,
        guidance_path_local: np.ndarray,
        spline_result: BSplineResult,
        mpc_local_xy: np.ndarray,
        robot_pose_world: np.ndarray,
        stage_timing_ms: dict[str, float],
        target_world: np.ndarray,
        clear_goal_world: np.ndarray | None,
        occupied_discs_world: list[DiscSpec],
        target_raw_local: np.ndarray,
        local_goal_raw_local: np.ndarray,
        local_goal_clamped_local: np.ndarray,
        local_goal_projected_local: np.ndarray,
        target_outside_local_map: bool,
        map_bundle,
        fallback_used: bool,
        v_ref: float,
        delta_t: float,
        d_j: float,
        observable_radius: float | None,
        include_edt_debug: bool,
        requested_local_map_window_size_m: float | None,
    ) -> dict:
        target_traj_world = self.map_adapter.local_to_world(robot_pose_world, guidance_path_local)
        astar_world = self.map_adapter.local_to_world(robot_pose_world, astar_local_xy)
        spline_world = self.map_adapter.local_to_world(robot_pose_world, spline_result.samples)
        mpc_world = self.map_adapter.local_to_world(robot_pose_world, mpc_local_xy)
        raw_world = self.map_adapter.local_to_world(robot_pose_world, np.asarray(target_raw_local, dtype=float).reshape(1, 2))[0]
        local_goal_raw_world = self.map_adapter.local_to_world(robot_pose_world, np.asarray(local_goal_raw_local, dtype=float).reshape(1, 2))[0]
        local_goal_clamped_world = self.map_adapter.local_to_world(robot_pose_world, np.asarray(local_goal_clamped_local, dtype=float).reshape(1, 2))[0]
        local_goal_projected_world = self.map_adapter.local_to_world(robot_pose_world, np.asarray(local_goal_projected_local, dtype=float).reshape(1, 2))[0]
        d_current = float(np.linalg.norm(robot_pose_world[:2] - target_world[:2]))
        x_min, x_max, y_min, y_max = self.map_adapter.map_limits(map_bundle)
        local_map_square_local = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ], dtype=float)
        local_map_square_world = self.map_adapter.local_to_world(robot_pose_world, local_map_square_local)
        local_map_occ_local = self.map_adapter.occupancy_to_local_points(map_bundle, stride=2)
        local_map_occ_world = self.map_adapter.local_to_world(robot_pose_world, local_map_occ_local) if len(local_map_occ_local) > 0 else np.empty((0, 2), dtype=float)
        local_map_edt_world = np.empty((0, 2), dtype=float)
        local_map_edt_values = np.empty((0,), dtype=float)
        if include_edt_debug:
            local_map_edt_local, local_map_edt_values = self.map_adapter.sample_edt_local_points(map_bundle, stride=1)
            if len(local_map_edt_local) > 0:
                local_map_edt_world = self.map_adapter.local_to_world(robot_pose_world, local_map_edt_local)
        observable_circle_world = np.empty((0, 2), dtype=float)
        if observable_radius is not None and float(observable_radius) > 0.0:
            observable_circle_local = self.map_adapter.circle_to_local_points(float(observable_radius))
            observable_circle_world = self.map_adapter.local_to_world(robot_pose_world, observable_circle_local)
        clear_goal_xy = None if clear_goal_world is None else np.asarray(clear_goal_world, dtype=float).reshape(-1)[:2]
        occupied_disc_debug = [
            {
                "center_world_xy": np.asarray(disc.center_world_xy, dtype=float).reshape(-1)[:2],
                "radius": float(disc.radius),
            }
            for disc in occupied_discs_world
        ]

        return {
            "arrive": False,
            "planning_success": bool(planning_success),
            "stage_timing_ms": {
                "hybrid_astar": float(stage_timing_ms.get("hybrid_astar", 0.0)),
                "hybrid_astar_search": float(stage_timing_ms.get("hybrid_astar_search", 0.0)),
                "fallback": float(stage_timing_ms.get("fallback", 0.0)),
                "bspline": float(stage_timing_ms.get("bspline", 0.0)),
                "mpc": float(stage_timing_ms.get("mpc", 0.0)),
            },
            "d_current": d_current,
            "d_desired": float(self.cfg.d_desired),
            "d_j": float(d_j),
            "v_ref": float(v_ref),
            "delta_t": float(delta_t),
            "fallback_used": bool(fallback_used),
            "planning_target_world": np.asarray(target_world[:2], dtype=float),
            "clear_goal_world": clear_goal_xy,
            "occupied_discs_world": occupied_disc_debug,
            "target_raw_local": np.asarray(target_raw_local, dtype=float),
            "local_goal_raw_local": np.asarray(local_goal_raw_local, dtype=float),
            "local_goal_clamped_local": np.asarray(local_goal_clamped_local, dtype=float),
            "local_goal_projected_local": np.asarray(local_goal_projected_local, dtype=float),
            "target_outside_local_map": bool(target_outside_local_map),
            "target_raw_world": np.asarray(raw_world, dtype=float),
            "local_goal_raw_world": np.asarray(local_goal_raw_world, dtype=float),
            "local_goal_clamped_world": np.asarray(local_goal_clamped_world, dtype=float),
            "local_goal_projected_world": np.asarray(local_goal_projected_world, dtype=float),
            "local_map_resolution": float(map_bundle.resolution),
            "local_map_shape": np.array([map_bundle.height, map_bundle.width], dtype=int),
            "local_map_extent_m": np.array([x_max - x_min, y_max - y_min], dtype=float),
            "local_map_window_size_m_requested": None if requested_local_map_window_size_m is None else float(requested_local_map_window_size_m),
            "local_map_square_path_list": self.map_adapter.to_plot_array(local_map_square_world),
            "local_map_occupancy_points": self.map_adapter.to_plot_array(local_map_occ_world),
            "observable_circle_path_list": self.map_adapter.to_plot_array(observable_circle_world),
            "observable_radius": 0.0 if observable_radius is None else float(observable_radius),
            "local_map_edt_points": self.map_adapter.to_plot_array(local_map_edt_world),
            "local_map_edt_values": np.asarray(local_map_edt_values, dtype=float),
            "show_edt": bool(include_edt_debug),
            "target_traj_path_list": self.map_adapter.to_plot_array(target_traj_world),
            "hybrid_astar_path_list": self.map_adapter.to_plot_array(astar_world),
            "bspline_path_list": self.map_adapter.to_plot_array(spline_world),
            "mpc_path_list": self.map_adapter.to_plot_array(mpc_world),
        }

    def _empty_debug_info(self, arrive: bool) -> dict:
        empty = np.empty((2, 0), dtype=float)
        zero_xy = np.zeros((2,), dtype=float)
        return {
            "arrive": bool(arrive),
            "planning_success": False,
            "stage_timing_ms": {
                "hybrid_astar": 0.0,
                "hybrid_astar_search": 0.0,
                "fallback": 0.0,
                "bspline": 0.0,
                "mpc": 0.0,
            },
            "d_current": 0.0,
            "d_desired": float(self.cfg.d_desired),
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
            "local_map_square_path_list": empty,
            "local_map_occupancy_points": empty,
            "observable_circle_path_list": empty,
            "observable_radius": 0.0,
            "local_map_edt_points": empty,
            "local_map_edt_values": np.empty((0,), dtype=float),
            "show_edt": False,
            "target_traj_path_list": empty,
            "hybrid_astar_path_list": empty,
            "bspline_path_list": empty,
            "mpc_path_list": empty,
        }
