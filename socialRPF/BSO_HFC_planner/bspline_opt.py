from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from .local_map import LidarLocalMapAdapter, LocalMapBundle


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def compute_path_headings(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)

    diffs = np.diff(points, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate((headings, headings[-1:]))
    return np.array([wrap_angle(angle) for angle in headings], dtype=float)


def resample_polyline_by_count(points: np.ndarray, count: int) -> np.ndarray:
    """Uniformly resample a polyline by arc length."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1 or count <= 1:
        return np.repeat(points[:1], max(count, 1), axis=0)

    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclen = np.concatenate(([0.0], np.cumsum(seg_len)))
    if arclen[-1] <= 1e-9:
        return np.repeat(points[:1], max(count, 1), axis=0)
    arclen = arclen / arclen[-1]

    target_t = np.linspace(0.0, 1.0, max(count, 1))
    x = np.interp(target_t, arclen, points[:, 0])
    y = np.interp(target_t, arclen, points[:, 1])
    return np.column_stack((x, y))


@dataclass
class BSplineConfig:
    """B-spline parameters, following the paper's notation where possible."""

    p: int
    num_ctrl_points: int
    num_samples: int
    optimize_maxiter: int
    d_thr: float
    omega_c: float
    omega_g: float
    omega_s: float
    omega_d: float
    v_max: float
    a_max: float


@dataclass
class BSplineResult:
    p: int
    control_points: np.ndarray
    knots: np.ndarray
    delta_t: float
    samples: np.ndarray
    sample_times: np.ndarray
    headings: np.ndarray


class BSplineOptimizer:
    """Optimize the paper's uniform B-spline objective on top of the Hybrid A* seed path."""

    def __init__(self, cfg: BSplineConfig) -> None:
        self.cfg = cfg

    def build_seed_control_points(self, astar_path_local: np.ndarray) -> np.ndarray:
        points = np.asarray(astar_path_local, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("astar_path_local must contain at least one path state.")
        if points.shape[1] >= 2:
            points = points[:, :2]

        ctrl_count = max(int(self.cfg.num_ctrl_points), int(self.cfg.p) + 2)
        return resample_polyline_by_count(points, ctrl_count)

    def build_guidance_points(self, guidance_path_local: np.ndarray) -> np.ndarray:
        points = np.asarray(guidance_path_local, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("guidance_path_local must contain at least one path state.")
        if points.shape[1] >= 2:
            points = points[:, :2]

        ctrl_count = max(int(self.cfg.num_ctrl_points), int(self.cfg.p) + 2)
        return resample_polyline_by_count(points, ctrl_count)

    def optimize(
        self,
        seed_control_points: np.ndarray,
        guidance_path_local: np.ndarray,
        map_bundle: LocalMapBundle,
        delta_t: float,
    ) -> BSplineResult:
        ctrl_pts = np.asarray(seed_control_points, dtype=float)
        p = int(self.cfg.p)
        if len(ctrl_pts) < p + 1:
            raise ValueError("Not enough control points for the configured B-spline order p.")

        guidance_points = self.build_guidance_points(guidance_path_local)
        fixed_prefix = ctrl_pts[:p]
        fixed_suffix = ctrl_pts[-p:]
        inner = ctrl_pts[p : len(ctrl_pts) - p]

        if inner.size == 0:
            samples, sample_times, headings = self.evaluate(ctrl_pts, delta_t)
            return BSplineResult(
                p=p,
                control_points=ctrl_pts,
                knots=self.make_open_uniform_knots(len(ctrl_pts), p, delta_t),
                delta_t=float(delta_t),
                samples=samples,
                sample_times=sample_times,
                headings=headings,
            )

        flat_inner = inner.reshape(-1)

        x_min, x_max, y_min, y_max = LidarLocalMapAdapter.map_limits(map_bundle)
        bounds = []
        for _ in range(inner.shape[0]):
            bounds.append((x_min, x_max))
            bounds.append((y_min, y_max))

        objective = lambda vars_flat: self._objective(
            vars_flat,
            fixed_prefix,
            fixed_suffix,
            guidance_points,
            map_bundle,
            delta_t,
        )
        result = minimize(
            objective,
            flat_inner,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(self.cfg.optimize_maxiter)},
        )

        if result.success and result.x is not None:
            ctrl_pts = self._reconstruct(result.x, fixed_prefix, fixed_suffix)

        samples, sample_times, headings = self.evaluate(ctrl_pts, delta_t)
        return BSplineResult(
            p=p,
            control_points=ctrl_pts,
            knots=self.make_open_uniform_knots(len(ctrl_pts), p, delta_t),
            delta_t=float(delta_t),
            samples=samples,
            sample_times=sample_times,
            headings=headings,
        )

    def evaluate(
        self,
        control_points: np.ndarray,
        delta_t: float,
        num_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        control_points = np.asarray(control_points, dtype=float)
        knots = self.make_open_uniform_knots(len(control_points), self.cfg.p, delta_t)
        sample_count = int(num_samples if num_samples is not None else self.cfg.num_samples)
        t_start = float(knots[self.cfg.p])
        t_end = float(knots[-self.cfg.p - 1])
        sample_times = np.linspace(t_start, t_end, max(sample_count, 2))
        spline_x = BSpline(knots, control_points[:, 0], self.cfg.p)(sample_times)
        spline_y = BSpline(knots, control_points[:, 1], self.cfg.p)(sample_times)
        samples = np.column_stack((spline_x, spline_y))
        headings = compute_path_headings(samples)
        return samples, sample_times, headings

    @staticmethod
    def make_open_uniform_knots(num_ctrl_points: int, p: int, delta_t: float) -> np.ndarray:
        if num_ctrl_points < p + 1:
            raise ValueError("num_ctrl_points must be at least p + 1")
        delta_t = max(float(delta_t), 1e-6)
        num_spans = num_ctrl_points - p
        if num_spans <= 0:
            raise ValueError("num_ctrl_points must exceed p")

        start = np.zeros((p + 1,), dtype=float)
        internal = delta_t * np.arange(1, num_spans, dtype=float)
        end = np.full((p + 1,), num_spans * delta_t, dtype=float)
        return np.concatenate((start, internal, end))

    @staticmethod
    def velocity(ctrl_pts: np.ndarray, delta_t: float) -> np.ndarray:
        return np.diff(ctrl_pts, axis=0) / max(float(delta_t), 1e-6)

    @staticmethod
    def acceleration(ctrl_pts: np.ndarray, delta_t: float) -> np.ndarray:
        vel = BSplineOptimizer.velocity(ctrl_pts, delta_t)
        return np.diff(vel, axis=0) / max(float(delta_t), 1e-6)

    @staticmethod
    def jerk(ctrl_pts: np.ndarray, delta_t: float) -> np.ndarray:
        acc = BSplineOptimizer.acceleration(ctrl_pts, delta_t)
        return np.diff(acc, axis=0) / max(float(delta_t), 1e-6)

    def _objective(
        self,
        flat_inner: np.ndarray,
        fixed_prefix: np.ndarray,
        fixed_suffix: np.ndarray,
        guidance_points: np.ndarray,
        map_bundle: LocalMapBundle,
        delta_t: float
    ) -> float:
        ctrl_pts = self._reconstruct(flat_inner, fixed_prefix, fixed_suffix)
        p = int(self.cfg.p)
        num_ctrl = len(ctrl_pts)

        core_slice = slice(p, num_ctrl - p)
        inner_ctrl = ctrl_pts[core_slice]
        inner_guidance = guidance_points[core_slice]

        collision_cost = self._collision_cost(inner_ctrl, map_bundle)
        guidance_cost = 0.0
        if len(inner_ctrl) > 0:
            diff = inner_ctrl - inner_guidance
            guidance_cost = float(np.sum(diff * diff))

        jerk = self.jerk(ctrl_pts, delta_t)
        jerk_window = jerk[max(p - 3, 0) : max(num_ctrl - p, 0)]
        smoothness_cost = float(np.sum(jerk_window * jerk_window)) if jerk_window.size else 0.0

        velocity = self.velocity(ctrl_pts, delta_t)
        velocity_window = velocity[max(p - 1, 0) : max(num_ctrl - p, 0)]
        velocity_cost = 0.0
        if velocity_window.size:
            speed_sq = np.sum(velocity_window * velocity_window, axis=1)
            penalty = np.maximum(0.0, speed_sq - float(self.cfg.v_max) ** 2)
            velocity_cost = float(np.sum(penalty * penalty))

        acceleration = self.acceleration(ctrl_pts, delta_t)
        acceleration_window = acceleration[max(p - 2, 0) : max(num_ctrl - p, 0)]
        acceleration_cost = 0.0
        if acceleration_window.size:
            accel_sq = np.sum(acceleration_window * acceleration_window, axis=1)
            penalty = np.maximum(0.0, accel_sq - float(self.cfg.a_max) ** 2)
            acceleration_cost = float(np.sum(penalty * penalty))

        return float(
            self.cfg.omega_c * collision_cost
            + self.cfg.omega_g * guidance_cost
            + self.cfg.omega_s * smoothness_cost
            + self.cfg.omega_d * (velocity_cost + acceleration_cost)
        )

    def _collision_cost(self, inner_ctrl_pts: np.ndarray, map_bundle: LocalMapBundle) -> float:
        points = np.asarray(inner_ctrl_pts, dtype=float)
        if points.size == 0:
            return 0.0

        distances = LidarLocalMapAdapter.sample_distances(map_bundle, points)
        penalties = np.maximum(0.0, float(self.cfg.d_thr) - distances)
        return float(np.sum(penalties * penalties))

    @staticmethod
    def _reconstruct(flat_inner: np.ndarray, fixed_prefix: np.ndarray, fixed_suffix: np.ndarray) -> np.ndarray:
        inner = np.asarray(flat_inner, dtype=float).reshape(-1, 2)
        return np.vstack((fixed_prefix, inner, fixed_suffix))
