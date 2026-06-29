#!/usr/bin/env python

import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import sobol_seq
from numba.core.errors import NumbaWarning
import warnings

try:
    import tf.transformations as tft
except ImportError:
    tft = None

try:
    from gctl.curve_generator import curve_generator
except ImportError:
    curve_generator = None

from traj_predictor import get_predictor

warnings.simplefilter("ignore", category=NumbaWarning)


position_dict = {
    "front": 0,
    "back": np.pi,
    "left_side": np.pi * 0.5,
    "right_side": np.pi * 1.5,
}

traj_predictor_configs = {
    "cv": {
        "name": "cv",
        "dt": 0.1,
        "history_length": 8,
        "prediction_horizon": 2.5,
    },
    "cvkf": {
        "name": "cvkf",
        "dt": 0.1,
        "prediction_horizon": 2.5,
        "history_length": 8,
        "predictor": {
            "num_samples": 20,
        },
    },
}


class MPCStats:
    def __init__(self):
        self.number = 0
        self.total_time = 0
        self.average_time = 0
        self.failures = 0
        self.per_step_stats = []

    def update_time(self, time_val, step_num):
        if time_val > 0:
            self.number += 1
            self.total_time += time_val
            self.average_time = self.total_time / self.number
            self.per_step_stats.append(
                {
                    "step": step_num,
                    "solution_time_ms": time_val * 1000,
                    "status": "success",
                }
            )

    def record_failure(self, step_num, error_msg=""):
        self.failures += 1
        self.per_step_stats.append(
            {
                "step": step_num,
                "status": "failure",
                "error": error_msg,
            }
        )

    def get_stats_dict(self):
        total = self.number + self.failures
        return {
            "average_solution_time_ms": 1000 * self.average_time if self.number > 0 else 0,
            "solver_failures": self.failures,
            "total_solutions_attempted": total,
            "success_rate": self.number / total if total > 0 else 0,
            "per_step_stats": self.per_step_stats,
        }


@nb.njit(cache=True)
def _project_points_to_equirectangular_jit(extrinsic_matrix, points_world, image_width, image_height):
    num_points = points_world.shape[0]
    projected_points = np.empty((num_points, 2), dtype=np.int32)
    valid_points_mask = np.zeros(num_points, dtype=np.bool_)

    for i in range(num_points):
        point_h = np.array([points_world[i, 0], points_world[i, 1], points_world[i, 2], 1.0])
        point_in_camera_h = extrinsic_matrix @ point_h
        x, y, z = point_in_camera_h[0], point_in_camera_h[1], point_in_camera_h[2]

        if z > 0:
            radius = np.sqrt(x**2 + y**2 + z**2)
            if radius > 1e-6:
                theta = np.arctan2(x, z)
                phi = np.arcsin(-y / radius)
                norm_theta = (theta + np.pi) / (2 * np.pi)
                norm_phi = (np.pi / 2 - phi) / np.pi
                projected_points[i, 0] = int(image_width * norm_theta)
                projected_points[i, 1] = int(image_height * norm_phi)
                valid_points_mask[i] = True

    return projected_points[valid_points_mask]


def calculate_bbox_from_cylinder_equirectangular(
    extrinsic_matrix,
    bottom_cylinder_center,
    img_size_wh,
    radius,
    height,
    wraparound_threshold=0.5,
):
    img_w, img_h = img_size_wh
    angles = np.linspace(0, 2 * np.pi, num=72)
    offsets_xy = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])
    offsets = np.hstack([offsets_xy, np.zeros((72, 1))])

    bottom_center_world = np.array(bottom_cylinder_center)
    top_center_world = bottom_center_world + np.array([0, 0, height])
    top_circle_points = top_center_world + offsets
    bottom_circle_points = bottom_center_world + offsets
    circle_points_world = np.vstack(
        [top_circle_points, bottom_circle_points, top_center_world, bottom_center_world]
    )

    image_points = _project_points_to_equirectangular_jit(
        extrinsic_matrix, circle_points_world, img_w, img_h
    )
    if image_points.shape[0] == 0:
        return [], None

    xmin_raw, ymin_raw = image_points.min(axis=0)
    xmax_raw, ymax_raw = image_points.max(axis=0)

    if (xmax_raw - xmin_raw) > img_w * wraparound_threshold:
        points_left = image_points[image_points[:, 0] < img_w / 2]
        points_right = image_points[image_points[:, 0] >= img_w / 2]
        bboxes = []
        if points_left.shape[0] > 0:
            xmin_l, ymin_l = points_left.min(axis=0)
            xmax_l, ymax_l = points_left.max(axis=0)
            bboxes.append(
                (
                    max(0, int(xmin_l)),
                    max(0, int(ymin_l)),
                    min(img_w - 1, int(xmax_l)),
                    min(img_h - 1, int(ymax_l)),
                )
            )
        if points_right.shape[0] > 0:
            xmin_r, ymin_r = points_right.min(axis=0)
            xmax_r, ymax_r = points_right.max(axis=0)
            bboxes.append(
                (
                    max(0, int(xmin_r)),
                    max(0, int(ymin_r)),
                    min(img_w - 1, int(xmax_r)),
                    min(img_h - 1, int(ymax_r)),
                )
            )
        return bboxes, image_points

    bbox = (
        max(0, int(xmin_raw)),
        max(0, int(ymin_raw)),
        min(img_w - 1, int(xmax_raw)),
        min(img_h - 1, int(ymax_raw)),
    )
    return [bbox], image_points


@nb.njit(cache=True)
def calculate_iou_jit(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0.0, inter_x_max - inter_x_min) * max(0.0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    if bbox1_area < 1e-6 or inter_area == 0.0:
        return 0.0

    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area


def get_cylinder_bottom_center(last_point, current_point, bias=0.0):
    direction = current_point - last_point
    angle = np.arctan2(direction[1], direction[0])
    return np.array(
        [
            current_point[0] + bias * np.cos(angle),
            current_point[1] + bias * np.sin(angle),
            0.0,
        ]
    )


def getTransformFromSamplePoint(sample_point, cylinder_bottom_center):
    theta = np.arctan2(
        cylinder_bottom_center[1] - sample_point[1],
        cylinder_bottom_center[0] - sample_point[0],
    )
    if tft is not None:
        t_world_base = tft.euler_matrix(0, 0, theta)
    else:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        t_world_base = np.array(
            [
                [cos_theta, -sin_theta, 0.0, 0.0],
                [sin_theta, cos_theta, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    t_world_base[0, 3] = sample_point[0]
    t_world_base[1, 3] = sample_point[1]
    t_base_world = np.linalg.inv(t_world_base)
    return theta, t_base_world


@nb.njit(cache=True)
def cal_dist_to_closest_point_jit(point, positions):
    num_positions = positions.shape[0]
    if num_positions == 0:
        return np.inf

    min_dist_sq = np.inf
    for i in range(num_positions):
        dx = point[0] - positions[i, 0]
        dy = point[1] - positions[i, 1]
        dist_sq = dx**2 + dy**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return np.sqrt(min_dist_sq)


def cal_cost_distanceToPerson(sample_point, predict_distracter_positions, max_dist=2.0):
    dist_to_person = cal_dist_to_closest_point_jit(sample_point, predict_distracter_positions)
    if dist_to_person > max_dist:
        return 0.0
    if dist_to_person < 1e-6:
        return 1000.0
    cost_to_person = (max_dist - dist_to_person) / (max_dist * dist_to_person)
    return cost_to_person**2


def generateSamplePoints_sobol(last_point, current_point, in_radius, out_radius, num_points):
    sobol_points = sobol_seq.i4_sobol_generate(2, num_points)
    radii = np.sqrt(in_radius**2 + (out_radius**2 - in_radius**2) * sobol_points[:, 0])
    angles = np.pi * sobol_points[:, 1]
    direction_vector = np.array(current_point) - np.array(last_point)
    direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
    angles += direction_angle + np.pi / 2
    points_x = radii * np.cos(angles) + current_point[0]
    points_y = radii * np.sin(angles) + current_point[1]
    return np.column_stack([points_x, points_y])


def getRobotGoal(target_person_pose, robot_pose, position, distance=2.0):
    if position not in position_dict:
        raise ValueError(f"Invalid position '{position}'. Expected one of: {list(position_dict.keys())}")

    target_person_pose = np.array(target_person_pose).squeeze()
    robot_pose = np.array(robot_pose).squeeze()

    if position == "front":
        yaw = target_person_pose[2]
        angle = yaw + position_dict["front"] + 2 * np.pi if yaw < 0 else yaw + position_dict["front"]
        x = target_person_pose[0] + np.cos(angle) * distance
        y = target_person_pose[1] + np.sin(angle) * distance
    elif position == "back":
        alpha_back = np.arctan2(
            target_person_pose[1] - robot_pose[1],
            target_person_pose[0] - robot_pose[0],
        )
        x = target_person_pose[0] - distance * np.cos(alpha_back)
        y = target_person_pose[1] - distance * np.sin(alpha_back)
        yaw = alpha_back
    elif position in ["left_side", "right_side"]:
        alpha_left = np.arctan2(
            np.sin(position_dict["left_side"] + target_person_pose[2]),
            np.cos(position_dict["left_side"] + target_person_pose[2]),
        )
        x_left = target_person_pose[0] + distance * np.cos(alpha_left)
        y_left = target_person_pose[1] + distance * np.sin(alpha_left)
        dist_left = np.linalg.norm(robot_pose[:2] - [x_left, y_left])

        alpha_right = np.arctan2(
            np.sin(position_dict["right_side"] + target_person_pose[2]),
            np.cos(position_dict["right_side"] + target_person_pose[2]),
        )
        x_right = target_person_pose[0] + distance * np.cos(alpha_right)
        y_right = target_person_pose[1] + distance * np.sin(alpha_right)
        dist_right = np.linalg.norm(robot_pose[:2] - [x_right, y_right])

        if position == "left_side":
            x, y = (x_left, y_left) if dist_left <= dist_right else (x_right, y_right)
        else:
            x, y = (x_right, y_right) if dist_left >= dist_right else (x_left, y_left)
        yaw = target_person_pose[2]
    else:
        print(f"Error: Unknown position '{position}'. Expected values are 'back','left_side', or 'right_side'.")

    return np.array([x, y, yaw])[:, np.newaxis]


def getTargetPersonState(env, yaw_target_list):
    target_person_vel = env.target.velocity
    target_person_pose = env.target.state
    v_target = np.sqrt(target_person_vel[0][0] ** 2 + target_person_vel[1][0] ** 2)
    yaw_target = np.arctan2(target_person_vel[1][0], target_person_vel[0][0])
    yaw_target_list.append(yaw_target)

    if len(yaw_target_list) > 1:
        w_target = np.gradient(yaw_target_list, env.step_time)[-1]
    else:
        w_target = 0.0

    target_person_vel = np.array([[v_target], [w_target]])
    target_person_pose[2] = yaw_target
    return target_person_vel, target_person_pose


class AdaptiveGoalGenerator:
    def __init__(self, predictor_type="cv", config=None):
        self.predictor_type = predictor_type
        if predictor_type not in traj_predictor_configs:
            raise ValueError(f"未支持的预测器类型: {predictor_type}")

        base_config = traj_predictor_configs[predictor_type].copy()
        config = config or {}
        for key, value in config.items():
            if key == "predictor" and "predictor" in base_config:
                base_config["predictor"].update(value)
            else:
                base_config[key] = value
        self.config = base_config

        self.dt = self.config.get("dt", 0.1)
        self.prediction_horizon = self.config.get("prediction_horizon", 2.0)
        self.history_length = self.config.get("history_length", 8)
        self.traj_predictor_params = {
            "name": self.predictor_type,
            "dt": self.dt,
            "prediction_horizon": self.prediction_horizon,
            "history_length": self.history_length,
            "predictor": self.config.get("predictor", {}),
        }
        self.predictor = get_predictor(self.predictor_type, self.traj_predictor_params)

        self.his_traj = {
            "updated_num": 0,
            "history_length": self.history_length,
            "dt": self.dt,
            "robot": deque(maxlen=self.history_length),
            "target": deque(maxlen=self.history_length),
            "humans": {},
        }
        self.predicted_traj = None
        self.goal_traj = None

        self.cg = curve_generator() if curve_generator is not None else None
        self.yaw_target_list = []
        self.T_optical_base = np.eye(4)
        self.T_optical_base[2, 3] = 0.5
        self.sample_points = None
        self.sample_costs = None
        self.best_sample_point = None

        self.current_goal_pose = None
        self.goal_update_counter = 0
        self.goal_update_interval = 3
        self.step_counter = 0
        self.takeover_pose_stats = MPCStats()

    def reset(self):
        self.his_traj = {
            "updated_num": 0,
            "history_length": self.history_length,
            "dt": self.dt,
            "robot": deque(maxlen=self.history_length),
            "target": deque(maxlen=self.history_length),
            "humans": {},
        }
        self.predicted_traj = None
        self.goal_traj = None
        self.current_goal_pose = None
        self.goal_update_counter = 0
        self.step_counter = 0
        self.takeover_pose_stats = MPCStats()

    def _get_grid_value(self, env, point):
        robot = getattr(env, "robot", None)
        if robot is None:
            return None
        if hasattr(robot, "get_grid_value"):
            return robot.get_grid_value(point)

        lidar = getattr(robot, "lidar", None)
        if lidar is None:
            return None
        if hasattr(lidar, "check_point_visible"):
            return lidar.check_point_visible(point)

        local_map = getattr(lidar, "local_map", None)
        if local_map is None or not hasattr(local_map, "map"):
            return None
        ego_state = getattr(lidar, "_state", getattr(robot, "state", None))
        if ego_state is None:
            return None

        ego_pose = np.asarray(ego_state).reshape(-1)[:3]
        point = np.asarray(point).reshape(-1)
        cos_theta = np.cos(ego_pose[2])
        sin_theta = np.sin(ego_pose[2])
        dx = point[0] - ego_pose[0]
        dy = point[1] - ego_pose[1]
        x_ego = cos_theta * dx + sin_theta * dy
        y_ego = -sin_theta * dx + cos_theta * dy
        x = int(round(x_ego / local_map.resolution + local_map.x_center))
        y = int(round(y_ego / local_map.resolution + local_map.y_center))
        if 0 <= x < local_map.map.shape[1] and 0 <= y < local_map.map.shape[0]:
            return local_map.map[y, x]
        return None

    def update_trajectory(self, env):
        for obj in env.objects:
            obj_vel = obj.velocity_xy
            obj_pose = obj.state
            point = (obj_pose[0, 0], obj_pose[1, 0], obj_vel[0, 0], obj_vel[1, 0])
            if obj.role == "robot":
                self.his_traj["robot"].append(point)

        visible_objs = env.robot.get_visible_objects()
        for obj in visible_objs:
            obj_vel = obj.velocity_xy
            obj_pose = obj.state
            point = (obj_pose[0, 0], obj_pose[1, 0], obj_vel[0, 0], obj_vel[1, 0])
            if obj.role == "target":
                self.his_traj["target"].append(point)
            elif obj.role == "human":
                if obj.id not in self.his_traj["humans"]:
                    self.his_traj["humans"][obj.id] = deque(maxlen=self.his_traj["history_length"])
                self.his_traj["humans"][obj.id].append(point)
        self.his_traj["updated_num"] += 1

    def predict_trajectory(self):
        num_agents = len(self.his_traj["humans"].keys()) + 2
        his_traj_arr = np.zeros((self.his_traj["history_length"], num_agents, 4))

        robot_hist_arr = np.array(self.his_traj["robot"])
        if len(robot_hist_arr) < self.history_length:
            padding_arr = np.tile(robot_hist_arr[0], (self.history_length - len(robot_hist_arr), 1))
            his_traj_arr[:, 0, :] = np.vstack([padding_arr, robot_hist_arr])
        else:
            his_traj_arr[:, 0, :] = robot_hist_arr

        target_hist_arr = np.array(self.his_traj["target"])
        if len(target_hist_arr) < self.history_length:
            padding_arr = np.tile(target_hist_arr[0], (self.history_length - len(target_hist_arr), 1))
            his_traj_arr[:, 1, :] = np.vstack([padding_arr, target_hist_arr])
        else:
            his_traj_arr[:, 1, :] = target_hist_arr

        for index, obj_id in enumerate(sorted(self.his_traj["humans"].keys())):
            human_history_arr = np.array(self.his_traj["humans"][obj_id])
            if len(human_history_arr) < self.history_length:
                padding_arr = np.tile(
                    human_history_arr[0],
                    (self.history_length - len(human_history_arr), 1),
                )
                his_traj_arr[:, index + 2, :] = np.vstack([padding_arr, human_history_arr])
            else:
                his_traj_arr[:, index + 2, :] = human_history_arr

        self.predicted_traj = self.predictor.predict(his_traj_arr)
        self.predicted_traj = np.squeeze(self.predicted_traj, axis=(0, 1))
        return self.predicted_traj

    def select_takeover_pose(self, env, n_sample_points=50, costmap_threshold=1.0):
        min_cost = 1000.0
        best_sample_point = None

        if len(self.his_traj["target"]) < 2:
            print("Warning: 历史轨迹不足，维持当前目标。")
            return self.current_goal_pose

        robot_position = np.array(self.his_traj["robot"][-1][:2])
        target_prev = np.array(self.his_traj["target"][-2][:2])
        target_curr = np.array(self.his_traj["target"][-1][:2])
        target_predictions = self.predicted_traj[:, 0, :]
        human_predictions = self.predicted_traj[:, 1:, :]

        sample_points = generateSamplePoints_sobol(
            target_prev,
            target_curr,
            in_radius=0.8,
            out_radius=3.0,
            num_points=n_sample_points,
        )
        self.sample_points = sample_points
        self.sample_costs = np.ones(len(sample_points)) * 1000.0

        w_dist_to_target = 10.0
        w_observability = 10.0
        w_travel_dist = 0.5
        w_to_person = 1.0
        w_stickiness = 1.0
        desired_follow_distance = 1.5
        robot_avg_speed = 1.0

        for i, sample_point in enumerate(sample_points):
            cost_in_map = self._get_grid_value(env, sample_point)
            if cost_in_map is None or cost_in_map >= costmap_threshold:
                continue

            dist_to_sample = np.linalg.norm(sample_point - robot_position)
            time_to_reach = dist_to_sample / robot_avg_speed
            future_index = int(time_to_reach / self.dt)
            future_index = min(future_index, human_predictions.shape[0] - 1)

            future_target_position = target_predictions[future_index, :2]
            future_human_positions = human_predictions[future_index, :, :2]
            prev_future_index = max(0, future_index - 1)
            prev_future_target_position = target_predictions[prev_future_index, :2]
            future_target_cylinder = get_cylinder_bottom_center(
                prev_future_target_position,
                future_target_position,
                bias=0.0,
            )

            prev_future_human_positions = human_predictions[prev_future_index, :, :2]
            future_human_cylinders = []
            if human_predictions.shape[1] > 0:
                for h_idx in range(human_predictions.shape[1]):
                    h_prev = prev_future_human_positions[h_idx]
                    h_curr = future_human_positions[h_idx]
                    if np.linalg.norm(h_curr - h_prev) < 0.01:
                        h_prev = h_curr - np.array([0.1, 0])
                    future_human_cylinders.append(get_cylinder_bottom_center(h_prev, h_curr, bias=0.0))

            cost_to_person = cal_cost_distanceToPerson(sample_point, future_human_positions, max_dist=2.5)

            total_cost_observability = 0.0
            yaw, t_base_world = getTransformFromSamplePoint(sample_point, future_target_cylinder[:2])
            extrinsic_matrix = self.T_optical_base @ t_base_world
            target_bboxes, _ = calculate_bbox_from_cylinder_equirectangular(
                extrinsic_matrix,
                future_target_cylinder,
                img_size_wh=(1920, 960),
                radius=0.4,
                height=1.7,
            )
            if not target_bboxes:
                continue

            if future_human_cylinders:
                occlusion_sum = 0.0
                for human_cylinder in future_human_cylinders:
                    human_bboxes, _ = calculate_bbox_from_cylinder_equirectangular(
                        extrinsic_matrix,
                        human_cylinder,
                        img_size_wh=(1920, 960),
                        radius=0.4,
                        height=1.7,
                    )
                    if human_bboxes:
                        for t_box in target_bboxes:
                            for h_box in human_bboxes:
                                occlusion_sum += calculate_iou_jit(t_box, h_box)
                total_cost_observability = occlusion_sum / len(future_human_cylinders)

            dist_to_target = np.linalg.norm(sample_point - target_curr)
            cost_target_dis = (dist_to_target - desired_follow_distance) ** 2
            cost_travel = dist_to_sample
            cost_stickiness = 0.0
            if self.current_goal_pose is not None:
                cost_stickiness = np.linalg.norm(sample_point - self.current_goal_pose[:2])

            cost_all = (
                w_dist_to_target * cost_target_dis
                + w_to_person * cost_to_person
                + w_observability * total_cost_observability
                + w_travel_dist * cost_travel
                + w_stickiness * cost_stickiness
            )
            self.sample_costs[i] = cost_all

            if cost_all < min_cost:
                min_cost = cost_all
                best_sample_point = np.array([sample_point[0], sample_point[1], 0.0])

        if best_sample_point is not None:
            final_target_idx = min(
                int(np.linalg.norm(best_sample_point[:2] - robot_position) / robot_avg_speed / self.dt),
                target_predictions.shape[0] - 1,
            )
            final_target_pos = target_predictions[final_target_idx, :2]
            final_yaw, _ = getTransformFromSamplePoint(best_sample_point[:2], final_target_pos)
            best_sample_point[2] = final_yaw

        self.best_sample_point = best_sample_point
        if best_sample_point is None and self.current_goal_pose is not None:
            print("INFO: 未找到更优的采样点，维持前一个目标。")
            return self.current_goal_pose
        return best_sample_point

    def _generate_linear_fallback(self, start_pose, goal_pose, steps=10):
        if steps <= 0:
            return [start_pose, goal_pose]
        path = []
        for i in range(steps + 1):
            alpha = i / float(steps)
            path.append((1 - alpha) * start_pose + alpha * goal_pose)
        return path

    def build_goal_trajectory(self, env, robot_pose, goal_params):
        start_t = time.time()
        self.update_trajectory(env)

        result = {
            "goal_traj": None,
            "predicted_traj": self.predicted_traj,
            "current_goal_pose": self.current_goal_pose,
            "visual_goal_traj": None,
            "mode": "adaptive",
            "success": True,
            "solve_time": 0.0,
        }

        if self.his_traj["updated_num"] < self.traj_predictor_params["history_length"]:
            print("INFO: 历史数据不足，使用简化的目标跟随逻辑。")
            target_person_vel, target_person_pose = getTargetPersonState(env, self.yaw_target_list)
            desired_pose = getRobotGoal(
                target_person_pose,
                robot_pose,
                goal_params["position"],
                goal_params["distance"],
            )
            if self.cg is not None:
                visual_goal_traj = self.cg.generate_curve("dubins", [robot_pose, desired_pose], 0.1, 5)
            else:
                visual_goal_traj = self._generate_linear_fallback(
                    np.array(robot_pose).reshape(3, 1),
                    np.array(desired_pose).reshape(3, 1),
                    steps=10,
                )
            goal_traj = np.array([p.squeeze()[:2] for p in visual_goal_traj], dtype=np.float32)
            result.update(
                {
                    "goal_traj": goal_traj,
                    "predicted_traj": None,
                    "visual_goal_traj": visual_goal_traj,
                    "mode": "fallback_history",
                    "solve_time": time.time() - start_t,
                }
            )
            return result

        self.predict_trajectory()
        self.goal_update_counter += 1

        needs_update = False
        if self.current_goal_pose is None:
            needs_update = True
            print("INFO: 无当前目标，选择初始接管点。")
        elif self.goal_update_counter >= self.goal_update_interval:
            needs_update = True
            print(f"INFO: 达到目标更新间隔 ({self.goal_update_interval} 步)，重新评估位置。")
        else:
            dist_to_goal = np.linalg.norm(robot_pose[:2].flatten() - self.current_goal_pose[:2].flatten())
            if dist_to_goal > (goal_params["distance"] * 2.5):
                needs_update = True
                print(f"INFO: 距离当前目标过远 (距离={dist_to_goal:.2f}m)，强制重新评估。")

        if needs_update:
            select_pose_start_t = time.time()
            try:
                optimal_pose = self.select_takeover_pose(env, n_sample_points=50)
                if optimal_pose is not None:
                    if self.current_goal_pose is not None:
                        change_dist = np.linalg.norm(self.current_goal_pose[:2] - optimal_pose[:2])
                        print(f"INFO: 更新最优位置。位置变化距离: {change_dist:.2f}m")
                    self.current_goal_pose = optimal_pose
                    self.goal_update_counter = 0
                else:
                    print("WARN: select_takeover_pose 未返回有效目标，维持原目标。")
                self.takeover_pose_stats.update_time(time.time() - select_pose_start_t, self.step_counter)
            except Exception as exc:
                print(f"select_takeover_pose failed: {exc}")
                self.takeover_pose_stats.record_failure(self.step_counter, str(exc))

        if self.current_goal_pose is not None and self.predicted_traj is not None:
            target_curr_pos = np.array(self.his_traj["target"][-1][:2])
            relative_vector = self.current_goal_pose[:2].flatten() - target_curr_pos
            target_predicted_path = self.predicted_traj[:, 0, :2]
            robot_goal_path = target_predicted_path + relative_vector
            goal_traj = robot_goal_path.astype(np.float32)

            vis_path = []
            for i in range(len(robot_goal_path)):
                if i < len(robot_goal_path) - 1:
                    diff = robot_goal_path[i + 1] - robot_goal_path[i]
                    yaw = np.arctan2(diff[1], diff[0])
                else:
                    if len(robot_goal_path) > 1:
                        diff = robot_goal_path[i] - robot_goal_path[i - 1]
                        yaw = np.arctan2(diff[1], diff[0])
                    else:
                        yaw = robot_pose[2, 0]
                vis_path.append(np.array([robot_goal_path[i, 0], robot_goal_path[i, 1], yaw]).reshape(3, 1))

            result.update(
                {
                    "goal_traj": goal_traj,
                    "predicted_traj": self.predicted_traj,
                    "current_goal_pose": self.current_goal_pose,
                    "visual_goal_traj": vis_path,
                    "mode": "adaptive",
                    "solve_time": time.time() - start_t,
                }
            )
            return result

        print("WARN: 没有有效的 current_goal_pose 或 predicted_traj，将以当前位置为目标。")
        horizon = max(1, int(self.prediction_horizon / self.dt))
        hold_goal = np.tile(robot_pose[:2].flatten(), (horizon, 1)).astype(np.float32)
        result.update(
            {
                "goal_traj": hold_goal,
                "predicted_traj": self.predicted_traj,
                "current_goal_pose": self.current_goal_pose,
                "visual_goal_traj": None,
                "mode": "hold_position",
                "solve_time": time.time() - start_t,
            }
        )
        return result

    def visualize_sample_points(self, env):
        if self.sample_points is None or self.sample_points.size == 0 or self.sample_costs is None:
            return

        sample_points_to_draw = []
        sample_colors = []
        valid_mask = self.sample_costs < 1000
        if np.any(valid_mask):
            min_cost = np.min(self.sample_costs[valid_mask])
            max_cost = np.max(self.sample_costs[valid_mask])
            cost_range = max(max_cost - min_cost, 1e-6)
            cmap = plt.colormaps["viridis"]

            for i, point in enumerate(self.sample_points):
                if self.sample_costs[i] < 1000:
                    sample_points_to_draw.append([point[0], point[1]])
                    norm_cost = (self.sample_costs[i] - min_cost) / cost_range
                    sample_colors.append(cmap(1 - norm_cost))

        if sample_points_to_draw:
            for point, color in zip(sample_points_to_draw, sample_colors):
                env.draw_points([point], s=30, c=color, refresh=True)
