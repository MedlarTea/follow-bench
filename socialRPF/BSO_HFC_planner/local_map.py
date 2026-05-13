from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class LocalMapBundle:
    """Planner-friendly view of ir-sim's robot-centric lidar occupancy map."""

    occupancy: np.ndarray
    edt: np.ndarray
    resolution: float
    width: int
    height: int
    row_center: int
    col_center: int


class LidarLocalMapAdapter:
    """
    Convert ir-sim's lidar local_map into the two layers BSO-HFC really needs:
    occupancy and EDT.

    The planner operates fully in the robot-local frame:
    - local x: robot forward
    - local y: robot left
    """

    def __init__(
        self,
        occ_threshold: float = 0.65,
        clear_target_margin: float = 0.15,
        unknown_is_occupied: bool = False,
    ) -> None:
        self.occ_threshold = float(occ_threshold)
        self.clear_target_margin = float(clear_target_margin)
        self.unknown_is_occupied = bool(unknown_is_occupied)

    def build(
        self,
        local_map,
        robot_pose: np.ndarray,
        clear_goal_pose_world: np.ndarray | None = None,
        clear_goal_radius: float | None = None,
        occupied_discs_world: list | None = None,
        observable_radius: float | None = None,
        window_size_m: float | None = None,
    ) -> LocalMapBundle:
        if local_map is None or not hasattr(local_map, "map"):
            raise ValueError("local_map is required and must expose a 'map' attribute.")

        raw_map = np.asarray(local_map.map, dtype=float)
        occupancy = raw_map >= self.occ_threshold
        if self.unknown_is_occupied:
            occupancy = np.logical_or(occupancy, np.isclose(raw_map, 0.5))

        resolution = float(local_map.resolution)
        occupancy = occupancy.astype(bool, copy=False)
        if observable_radius is not None and float(observable_radius) > 0.0:
            occupancy = self.crop_inscribed_square_occupancy(
                occupancy,
                resolution,
                float(observable_radius),
                window_size_m=window_size_m,
            )

        bundle = self._make_bundle_from_occupancy(occupancy, resolution)

        if clear_goal_pose_world is not None:
            goal_xy = np.asarray(clear_goal_pose_world, dtype=float).reshape(-1)[:2][None, :]
            goal_local = self.world_to_local(robot_pose, goal_xy)[0]
            clear_radius = max(float(clear_goal_radius or 0.0), 0.0) + max(self.clear_target_margin, 0.0)
            if self._should_clear_goal_target(goal_local, clear_radius, bundle, observable_radius):
                cleared = self._clear_goal_connected_component(
                    bundle.occupancy,
                    bundle.resolution,
                    bundle.row_center,
                    bundle.col_center,
                    goal_local,
                    clear_radius,
                )
                if not cleared:
                    self._clear_disc(
                        bundle.occupancy,
                        bundle.resolution,
                        bundle.row_center,
                        bundle.col_center,
                        goal_local,
                        clear_radius,
                    )

        for disc in occupied_discs_world or []:
            disc_local, disc_radius = self._coerce_disc_spec(robot_pose, disc)
            if disc_local is None or disc_radius <= 0.0:
                continue
            self._occupy_disc(
                bundle.occupancy,
                bundle.resolution,
                bundle.row_center,
                bundle.col_center,
                disc_local,
                disc_radius,
            )

        bundle.edt = distance_transform_edt(~bundle.occupancy) * bundle.resolution
        return bundle

    @staticmethod
    def crop_inscribed_square_occupancy(
        occupancy: np.ndarray,
        resolution: float,
        observable_radius: float,
        window_size_m: float | None = None,
    ) -> np.ndarray:
        occ = np.asarray(occupancy, dtype=bool)
        height, width = occ.shape
        row_center = height // 2
        col_center = width // 2
        max_half_window = max(float(observable_radius) / np.sqrt(2.0), resolution)
        if window_size_m is None:
            half_window = max_half_window
        else:
            half_window = min(max(float(window_size_m) * 0.5, resolution), max_half_window)
        half_cells = max(int(np.floor(half_window / max(resolution, 1e-6))), 1)

        row_min = max(row_center - half_cells, 0)
        row_max = min(row_center + half_cells + 1, height)
        col_min = max(col_center - half_cells, 0)
        col_max = min(col_center + half_cells + 1, width)
        return occ[row_min:row_max, col_min:col_max].copy()

    def world_to_local(self, robot_pose: np.ndarray, points_world: np.ndarray) -> np.ndarray:
        pose = np.asarray(robot_pose, dtype=float).reshape(-1)
        points = np.asarray(points_world, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        dx = points[:, 0] - pose[0]
        dy = points[:, 1] - pose[1]
        cos_yaw = np.cos(pose[2])
        sin_yaw = np.sin(pose[2])
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy
        return np.column_stack((x_local, y_local))

    def local_to_world(self, robot_pose: np.ndarray, points_local: np.ndarray) -> np.ndarray:
        pose = np.asarray(robot_pose, dtype=float).reshape(-1)
        points = np.asarray(points_local, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        cos_yaw = np.cos(pose[2])
        sin_yaw = np.sin(pose[2])
        x_world = pose[0] + cos_yaw * points[:, 0] - sin_yaw * points[:, 1]
        y_world = pose[1] + sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
        return np.column_stack((x_world, y_world))

    @staticmethod
    def local_to_grid(map_bundle: LocalMapBundle, x_local: float, y_local: float) -> tuple[int, int]:
        row = int(round(float(y_local) / map_bundle.resolution + map_bundle.row_center))
        col = int(round(float(x_local) / map_bundle.resolution + map_bundle.col_center))
        return row, col

    @staticmethod
    def local_to_grid_float(map_bundle: LocalMapBundle, x_local: float, y_local: float) -> tuple[float, float]:
        row = float(y_local) / map_bundle.resolution + map_bundle.row_center
        col = float(x_local) / map_bundle.resolution + map_bundle.col_center
        return row, col

    @staticmethod
    def in_bounds(map_bundle: LocalMapBundle, x_local: float, y_local: float) -> bool:
        row, col = LidarLocalMapAdapter.local_to_grid(map_bundle, x_local, y_local)
        return 0 <= row < map_bundle.height and 0 <= col < map_bundle.width

    @staticmethod
    def sample_distance_bilinear(map_bundle: LocalMapBundle, x_local: float, y_local: float) -> float:
        row_f, col_f = LidarLocalMapAdapter.local_to_grid_float(map_bundle, x_local, y_local)
        if row_f < 0.0 or row_f > map_bundle.height - 1 or col_f < 0.0 or col_f > map_bundle.width - 1:
            return 0.0

        row0 = int(np.floor(row_f))
        col0 = int(np.floor(col_f))
        row1 = min(row0 + 1, map_bundle.height - 1)
        col1 = min(col0 + 1, map_bundle.width - 1)
        wy = row_f - row0
        wx = col_f - col0

        v00 = float(map_bundle.edt[row0, col0])
        v01 = float(map_bundle.edt[row0, col1])
        v10 = float(map_bundle.edt[row1, col0])
        v11 = float(map_bundle.edt[row1, col1])

        return float(
            (1.0 - wy) * (1.0 - wx) * v00
            + (1.0 - wy) * wx * v01
            + wy * (1.0 - wx) * v10
            + wy * wx * v11
        )

    @staticmethod
    def sample_distances(map_bundle: LocalMapBundle, points_local: np.ndarray) -> np.ndarray:
        points = np.asarray(points_local, dtype=float)
        if points.size == 0:
            return np.empty((0,), dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        row_f = points[:, 1] / map_bundle.resolution + map_bundle.row_center
        col_f = points[:, 0] / map_bundle.resolution + map_bundle.col_center
        distances = np.zeros((len(points),), dtype=float)

        valid = (row_f >= 0.0) & (row_f <= map_bundle.height - 1) & (col_f >= 0.0) & (col_f <= map_bundle.width - 1)
        if not np.any(valid):
            return distances

        row_f = row_f[valid]
        col_f = col_f[valid]
        row0 = np.floor(row_f).astype(int)
        col0 = np.floor(col_f).astype(int)
        row1 = np.minimum(row0 + 1, map_bundle.height - 1)
        col1 = np.minimum(col0 + 1, map_bundle.width - 1)

        wy = row_f - row0
        wx = col_f - col0

        edt = map_bundle.edt
        v00 = edt[row0, col0]
        v01 = edt[row0, col1]
        v10 = edt[row1, col0]
        v11 = edt[row1, col1]

        distances[valid] = (
            (1.0 - wy) * (1.0 - wx) * v00
            + (1.0 - wy) * wx * v01
            + wy * (1.0 - wx) * v10
            + wy * wx * v11
        )
        return distances.astype(float, copy=False)

    @staticmethod
    def map_limits(map_bundle: LocalMapBundle) -> tuple[float, float, float, float]:
        x_min = -map_bundle.col_center * map_bundle.resolution
        x_max = (map_bundle.width - map_bundle.col_center - 1) * map_bundle.resolution
        y_min = -map_bundle.row_center * map_bundle.resolution
        y_max = (map_bundle.height - map_bundle.row_center - 1) * map_bundle.resolution
        return x_min, x_max, y_min, y_max

    @staticmethod
    def to_plot_array(points_world: np.ndarray | list) -> np.ndarray:
        points = np.asarray(points_world, dtype=float)
        if points.size == 0:
            return np.empty((2, 0), dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return points[:, :2].T

    @staticmethod
    def occupancy_to_local_points(map_bundle: LocalMapBundle, stride: int = 1) -> np.ndarray:
        stride = max(int(stride), 1)
        rows, cols = np.nonzero(map_bundle.occupancy)
        if len(rows) == 0:
            return np.empty((0, 2), dtype=float)
        rows = rows[::stride]
        cols = cols[::stride]
        x_local = (cols - map_bundle.col_center) * map_bundle.resolution
        y_local = (rows - map_bundle.row_center) * map_bundle.resolution
        return np.column_stack((x_local, y_local)).astype(float, copy=False)

    @staticmethod
    def sample_edt_local_points(map_bundle: LocalMapBundle, stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
        stride = max(int(stride), 1)
        row_idx = np.arange(0, map_bundle.height, stride, dtype=int)
        col_idx = np.arange(0, map_bundle.width, stride, dtype=int)
        rows, cols = np.meshgrid(row_idx, col_idx, indexing="ij")
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        free_mask = ~map_bundle.occupancy[rows, cols]
        if not np.any(free_mask):
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

        rows = rows[free_mask]
        cols = cols[free_mask]
        x_local = (cols - map_bundle.col_center) * map_bundle.resolution
        y_local = (rows - map_bundle.row_center) * map_bundle.resolution
        points_local = np.column_stack((x_local, y_local)).astype(float, copy=False)
        values = map_bundle.edt[rows, cols].astype(float, copy=False)
        return points_local, values

    @staticmethod
    def circle_to_local_points(radius: float, num_points: int = 100) -> np.ndarray:
        radius = max(float(radius), 0.0)
        if radius <= 0.0:
            return np.empty((0, 2), dtype=float)
        num = max(int(num_points), 12)
        angles = np.linspace(0.0, 2.0 * np.pi, num, endpoint=True, dtype=float)
        x_local = radius * np.cos(angles)
        y_local = radius * np.sin(angles)
        return np.column_stack((x_local, y_local)).astype(float, copy=False)

    def _coerce_disc_spec(self, robot_pose: np.ndarray, disc) -> tuple[np.ndarray | None, float]:
        if disc is None:
            return None, 0.0

        if hasattr(disc, "center_world_xy") and hasattr(disc, "radius"):
            center_world = np.asarray(getattr(disc, "center_world_xy"), dtype=float).reshape(-1)[:2]
            radius = float(getattr(disc, "radius"))
        elif isinstance(disc, dict):
            center_world = np.asarray(disc.get("center_world_xy", disc.get("center_world")), dtype=float).reshape(-1)[:2]
            radius = float(disc.get("radius", 0.0))
        elif isinstance(disc, (tuple, list)) and len(disc) >= 2:
            center_world = np.asarray(disc[0], dtype=float).reshape(-1)[:2]
            radius = float(disc[1])
        else:
            raise ValueError("occupied_discs_world entries must expose center_world_xy and radius.")

        center_local = self.world_to_local(robot_pose, center_world[None, :])[0]
        return center_local, max(radius, 0.0)

    def _should_clear_goal_target(
        self,
        goal_local: np.ndarray,
        clear_radius: float,
        map_bundle: LocalMapBundle,
        observable_radius: float | None,
    ) -> bool:
        if clear_radius <= 0.0:
            return False
        goal = np.asarray(goal_local, dtype=float).reshape(-1)[:2]
        if not self.in_bounds(map_bundle, goal[0], goal[1]):
            return False

        margin = max(float(clear_radius), float(map_bundle.resolution))
        x_min, x_max, y_min, y_max = self.map_limits(map_bundle)
        if not (x_min + margin <= goal[0] <= x_max - margin and y_min + margin <= goal[1] <= y_max - margin):
            return False

        if observable_radius is not None and float(observable_radius) > 0.0:
            if float(np.linalg.norm(goal)) > float(observable_radius) - margin:
                return False
        return True

    def _clear_goal_connected_component(
        self,
        occupancy: np.ndarray,
        resolution: float,
        row_center: int,
        col_center: int,
        goal_local: np.ndarray,
        clear_radius: float,
    ) -> bool:
        search_radius = max(float(clear_radius), float(resolution))
        seed = self._find_goal_blob_seed(occupancy, resolution, row_center, col_center, goal_local, search_radius)
        if seed is None:
            return False

        max_blob_radius = max(float(clear_radius) * 1.75, float(clear_radius) + 2.0 * float(resolution))
        radius_cells = max(int(np.ceil(max_blob_radius / max(resolution, 1e-6))), 1)
        max_cells = max(24, int(np.ceil(np.pi * radius_cells * radius_cells * 1.5)))
        component_mask = self._extract_connected_component(occupancy, seed, radius_cells, max_cells)
        if component_mask is None or not np.any(component_mask):
            return False

        occupancy[component_mask] = False
        return True

    @staticmethod
    def _find_goal_blob_seed(
        occupancy: np.ndarray,
        resolution: float,
        row_center: int,
        col_center: int,
        goal_local: np.ndarray,
        search_radius: float,
    ) -> tuple[int, int] | None:
        row = int(round(float(goal_local[1]) / resolution + row_center))
        col = int(round(float(goal_local[0]) / resolution + col_center))
        if not (0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1]):
            return None

        radius_cells = max(int(np.ceil(max(search_radius, 0.0) / max(resolution, 1e-6))), 1)
        row_min = max(row - radius_cells, 0)
        row_max = min(row + radius_cells + 1, occupancy.shape[0])
        col_min = max(col - radius_cells, 0)
        col_max = min(col + radius_cells + 1, occupancy.shape[1])
        window = occupancy[row_min:row_max, col_min:col_max]
        occ_rows, occ_cols = np.nonzero(window)
        if len(occ_rows) == 0:
            return None

        occ_rows = occ_rows + row_min
        occ_cols = occ_cols + col_min
        dist_sq = (occ_rows - row) ** 2 + (occ_cols - col) ** 2
        best_idx = int(np.argmin(dist_sq))
        return int(occ_rows[best_idx]), int(occ_cols[best_idx])

    @staticmethod
    def _extract_connected_component(
        occupancy: np.ndarray,
        seed: tuple[int, int],
        max_radius_cells: int,
        max_cells: int,
    ) -> np.ndarray | None:
        seed_row, seed_col = int(seed[0]), int(seed[1])
        if not occupancy[seed_row, seed_col]:
            return None

        visited = np.zeros_like(occupancy, dtype=bool)
        component_mask = np.zeros_like(occupancy, dtype=bool)
        queue = deque([(seed_row, seed_col)])
        visited[seed_row, seed_col] = True
        count = 0

        while queue:
            row, col = queue.popleft()
            if (row - seed_row) ** 2 + (col - seed_col) ** 2 > max_radius_cells ** 2:
                continue
            if not occupancy[row, col]:
                continue

            component_mask[row, col] = True
            count += 1
            if count > max_cells:
                return None

            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    nxt_row = row + d_row
                    nxt_col = col + d_col
                    if 0 <= nxt_row < occupancy.shape[0] and 0 <= nxt_col < occupancy.shape[1] and not visited[nxt_row, nxt_col]:
                        visited[nxt_row, nxt_col] = True
                        queue.append((nxt_row, nxt_col))

        return component_mask if count > 0 else None

    @staticmethod
    def _make_bundle_from_occupancy(occupancy: np.ndarray, resolution: float) -> LocalMapBundle:
        occ = np.asarray(occupancy, dtype=bool)
        height, width = occ.shape
        row_center = height // 2
        col_center = width // 2
        edt = distance_transform_edt(~occ) * resolution
        return LocalMapBundle(
            occupancy=occ.copy(),
            edt=edt,
            resolution=float(resolution),
            width=width,
            height=height,
            row_center=row_center,
            col_center=col_center,
        )

    @staticmethod
    def _clear_disc(
        occupancy: np.ndarray,
        resolution: float,
        row_center: int,
        col_center: int,
        center_local: np.ndarray,
        radius: float,
    ) -> None:
        LidarLocalMapAdapter._apply_disc(
            occupancy,
            resolution,
            row_center,
            col_center,
            center_local,
            radius,
            occupied=False,
        )

    @staticmethod
    def _occupy_disc(
        occupancy: np.ndarray,
        resolution: float,
        row_center: int,
        col_center: int,
        center_local: np.ndarray,
        radius: float,
    ) -> None:
        LidarLocalMapAdapter._apply_disc(
            occupancy,
            resolution,
            row_center,
            col_center,
            center_local,
            radius,
            occupied=True,
        )

    @staticmethod
    def _apply_disc(
        occupancy: np.ndarray,
        resolution: float,
        row_center: int,
        col_center: int,
        center_local: np.ndarray,
        radius: float,
        occupied: bool,
    ) -> None:
        radius_cells = int(np.ceil(max(radius, 0.0) / max(resolution, 1e-6)))
        row = int(round(float(center_local[1]) / resolution + row_center))
        col = int(round(float(center_local[0]) / resolution + col_center))

        row_min = max(row - radius_cells, 0)
        row_max = min(row + radius_cells + 1, occupancy.shape[0])
        col_min = max(col - radius_cells, 0)
        col_max = min(col + radius_cells + 1, occupancy.shape[1])
        if row_min >= row_max or col_min >= col_max:
            return

        rr = np.arange(row_min, row_max)[:, None]
        cc = np.arange(col_min, col_max)[None, :]
        dr = rr - row
        dc = cc - col
        mask = (dr * dr + dc * dc) * (resolution ** 2) <= radius ** 2
        occupancy[row_min:row_max, col_min:col_max][mask] = bool(occupied)
