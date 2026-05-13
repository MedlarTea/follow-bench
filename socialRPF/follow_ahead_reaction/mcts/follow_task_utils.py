import math

import numpy as np


MODE_OFFSETS = {
    "front": 0.0,
    "back": np.pi,
    "left_side": np.pi / 2.0,
    "right_side": -np.pi / 2.0,
}

OBS_MODES = {"relative_pose", "task_error"}


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def format_distance_tag(distance):
    return f"{float(distance):.1f}".replace(".", "p")


def model_variant_dir(obs_mode):
    if obs_mode == "task_error":
        return "follow_task_models_task_error"
    return "follow_task_models"


def desired_local_point(mode, desired_distance):
    if mode == "back":
        return -float(desired_distance), 0.0
    if mode == "left_side":
        return 0.0, float(desired_distance)
    if mode == "right_side":
        return 0.0, -float(desired_distance)
    return float(desired_distance), 0.0


def desired_beta(human_yaw, mode):
    return wrap_to_pi(human_yaw + MODE_OFFSETS.get(mode, 0.0))


def direction_diff_deg(rel_vec, human_yaw, mode):
    beta = math.atan2(rel_vec[1], rel_vec[0])
    diff = abs(wrap_to_pi(desired_beta(human_yaw, mode) - beta))
    return diff * 180.0 / np.pi


def local_diagnostics(rel_vec, human_yaw, robot_yaw, mode, desired_distance):
    heading_vec = np.array([np.cos(human_yaw), np.sin(human_yaw)], dtype=np.float64)
    lateral_vec = np.array([-np.sin(human_yaw), np.cos(human_yaw)], dtype=np.float64)
    lon = float(np.dot(rel_vec, heading_vec))
    lat = float(np.dot(rel_vec, lateral_vec))
    desired_lon, desired_lat = desired_local_point(mode, desired_distance)
    lon_err = lon - desired_lon
    lat_err = lat - desired_lat
    distance = float(np.linalg.norm(rel_vec))
    diff = direction_diff_deg(rel_vec, human_yaw, mode)
    yaw_err = abs(wrap_to_pi(robot_yaw - human_yaw))
    return {
        "distance": distance,
        "diff": diff,
        "lon": lon,
        "lat": lat,
        "lon_err": lon_err,
        "lat_err": lat_err,
        "yaw_err": yaw_err,
        "desired_lon": desired_lon,
        "desired_lat": desired_lat,
    }


def distance_reward(distance, desired_distance):
    desired_distance = float(desired_distance)
    min_safe = max(0.25, desired_distance - 1.25)
    near_low = max(0.25, desired_distance - 0.5)
    near_high = desired_distance + 0.5
    far_limit = desired_distance + 3.5

    if distance > far_limit or distance < min_safe:
        return -1.0
    if min_safe < distance < near_low:
        return distance - near_low
    if near_low <= distance <= near_high:
        return 0.5 * (0.5 - abs(distance - desired_distance))
    if near_high < distance <= far_limit:
        return -0.25 * (distance - near_low)
    return 0.0


def follow_reward(mode, desired_distance, rel_vec, human_yaw, robot_yaw):
    diag = local_diagnostics(rel_vec, human_yaw, robot_yaw, mode, desired_distance)

    r_d = distance_reward(diag["distance"], desired_distance)
    r_d = r_d / 2.0 + 0.500000001
    r_o = (25.0 - diag["diff"]) / 25.0

    scale = max(float(desired_distance), 1.0)
    shaping = 0.0
    shaping -= 0.20 * min(abs(diag["lon_err"]) / scale, 2.0)
    shaping -= 0.20 * min(abs(diag["lat_err"]) / scale, 2.0)
    shaping += 0.08 * (1.0 - min(diag["yaw_err"] / (np.pi / 2.0), 2.0))

    near_limit = max(0.5, float(desired_distance) - 0.7)
    if diag["distance"] < near_limit:
        denom = max(near_limit, 1e-6)
        shaping -= 0.55 * min((near_limit - diag["distance"]) / denom, 1.5)

    if (
        abs(diag["lon_err"]) < 0.35 * scale
        and abs(diag["lat_err"]) < 0.35 * scale
        and abs(diag["distance"] - float(desired_distance)) < 0.5
        and diag["diff"] < 25.0
    ):
        shaping += 0.25

    reward = r_d + r_o + shaping
    return reward, {
        "r_d": r_d,
        "r_o": r_o,
        "shaping": shaping,
        **diag,
    }


def build_rl_observation(rel_vec, human_yaw, robot_yaw, mode, desired_distance, obs_mode="relative_pose"):
    yaw_err = wrap_to_pi(robot_yaw - human_yaw)
    if obs_mode == "task_error":
        diag = local_diagnostics(rel_vec, human_yaw, robot_yaw, mode, desired_distance)
        return np.array([diag["lon_err"], diag["lat_err"], yaw_err], dtype=np.float32)

    return np.array([rel_vec[0], rel_vec[1], yaw_err], dtype=np.float32)
