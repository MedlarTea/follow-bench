import numpy as np
from collections import deque
position_dict = {
    "front": 0,
    "back": np.pi,
    "left_side": np.pi * 0.5,
    "right_side": np.pi * 1.5
}

traj_predictors = {
    "cv": {
        "name": "cv",
        "dt": 0.1,
        "history_length": 8,
        "prediction_horizon": 2.0  # s
    },
    "cvkf": {
        "name": "cvkf",
        "dt": 0.1,
        "prediction_horizon": 2.0,
        "history_length": 1,
        "predictor": {
            "num_samples": 20,
        }
    },
    "sgan": {
        "name": "sgan",
        "dt": 0.1,
        "prediction_horizon": 2.0,
        "history_length": 8,
        "predictor": {
            "path": 'sgan.pt',
            "use_gpu": False,
            "num_samples": 1,
            "deviation_penalty": True,
            "use_sgan_action": False,
            "use_sgan_mode": True,
        }
    }
}

## for dynamic search field
gamma = 0.1
sigma = 1.0   
max_range = 5.0  
fov_angle = 360

def update_trajectory(env, his_traj):
    """Only update visible objects' trajectory"""
    visible_obj_ids = [obj.id for obj in env.visible_object_list]
    for obj in env.objects:
        obj_vel = obj.velocity_xy
        obj_pose = obj.state
        point = (obj_pose[0, 0], obj_pose[1, 0], obj_vel[0, 0], obj_vel[1, 0])
        if obj.role == "robot":
            his_traj["robot"].append(point)
        elif obj.role == "target":
            his_traj["target"].append(point)
        if obj.role == "human" and obj.id in visible_obj_ids:
            if obj.id not in his_traj["humans"]:
                his_traj["humans"][obj.id] = deque(maxlen=his_traj["history_length"])
            his_traj["humans"][obj.id].append(point)
    his_traj["updated_num"] += 1
    # return his_traj

def predict_trajectory(traj_predictor, his_traj):
    his_traj_arr = np.zeros((his_traj["history_length"], len(his_traj["humans"].keys())+2, 4))  # (his_length, N, 4)
    # print("his_traj_arr shape:", his_traj_arr.shape)

    his_traj_arr[:, 0, :] = np.array(his_traj["robot"])
    his_traj_arr[:, 1, :] = np.array(his_traj["target"])
    for index, obj_id in enumerate(sorted(his_traj["humans"].keys())):
        his_traj_arr[:, index+2, :] = np.array(his_traj["humans"][obj_id])

    predicted_traj = traj_predictor.predict(his_traj_arr)
    predicted_traj = np.squeeze(predicted_traj, axis=(0, 1))

    # print("predicted_traj shape:", predicted_traj.shape)

    target_traj = predicted_traj[:, 0, :]  # (future_steps, 4)
    humans_traj = predicted_traj[:, 1:, :] if predicted_traj.shape[1] > 1 else []
    # print(humans_traj.shape)
    # print(len(sorted(his_traj["humans"].keys())))
    # predictions = traj_predictor.predict(his_traj["robot"][-1])
    predicted_human_traj = {}
    sorted_human_ids = sorted(his_traj["humans"].keys())
    if len(humans_traj) != 0:
        for index in range(humans_traj.shape[1]):
            obj_id = sorted_human_ids[index]
            # print(index, obj_id)
            predicted_human_traj[obj_id] = humans_traj[:, index, :]

    return target_traj, predicted_human_traj

def get_predicted_target_pose(predicted_traj):
    target_pose = np.zeros((3, 1))
    target_pose[0:2] = predicted_traj[-1, :2].reshape(2, 1)  # Get the predicted position
    last_position = predicted_traj[-2, :2].reshape(2, 1)
    yaw = np.arctan2(target_pose[1, 0] - last_position[1, 0], target_pose[0, 0] - last_position[0, 0])
    target_pose[2, 0] = yaw
    return target_pose

def get_yaw(last_position, current_position):
    """
    Calculate the yaw angle from last position to current position.
    :param last_position: Last position (x, y)
    :param current_position: Current position (x, y)
    :return: Yaw angle in radians
    """
    direction_vector = np.array(current_position) - np.array(last_position)
    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    return yaw

def korobov_lattice(n, dim, a):
    """Generate Korobov lattice points."""
    k = np.arange(1, n + 1)
    points = (np.outer(k, a) % n) / n
    return points

def generateSamplePoints_korobov_lattice(last_point, current_point, in_radius, out_radius, num_points):
    """
    Generate points in a semicircle space based on the direction from last_point to current_point using Korobov lattice
        :param last_point: The last position (x, y)
        :param current_point: The current position (x, y)
        :param in_radius: The inner radius of the sampling space
        :param out_radius: The outer radius of the sampling space
        :param num_points: Number of points to generate
        :return: List of generated points (x, y)
    """
    a = [1, 7]
    lattice_points = korobov_lattice(num_points, 2, a)
    
    # Map the Korobov lattice points to the radial range [in_radius, out_radius]
    radii = np.sqrt(in_radius**2 + (out_radius**2 - in_radius**2) * lattice_points[:, 0])
    
    # Map the Korobov lattice points to the angular range [0, pi]
    angles = np.pi * lattice_points[:, 1]
    
    # Compute the heading vector and its angle
    direction_vector = np.array(current_point) - np.array(last_point)
    direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
    
    # Rotate the points into the forward semicircle
    angles += (direction_angle - np.pi / 2)
    
    # Convert back to Cartesian coordinates
    points_x = radii * np.cos(angles) + current_point[0]
    points_y = radii * np.sin(angles) + current_point[1]
    
    points = []
    for x, y in zip(points_x, points_y):
        points.append([x, y])
    
    return points