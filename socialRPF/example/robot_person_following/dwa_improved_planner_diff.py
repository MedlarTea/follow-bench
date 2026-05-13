import os
import sys

if "--headless" in sys.argv:
    os.environ["MPLBACKEND"] = "Agg"

import irsim
import numpy as np
import argparse
import cv2
import time

# Repo root contains the RDA_planner and traj_predictor packages (example/robot_person_following -> ../..).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
    
from DWA_planner.dwa_improve import DWA
from collections import namedtuple
from sklearn.cluster import DBSCAN
from gctl.curve_generator import curve_generator
from traj_predictor import get_predictor
from collections import deque
from global_params import traj_predictors, position_dict, predict_trajectory, update_trajectory, get_predicted_target_pose
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import logging
# Show only ERROR-level and above log messages
logging.basicConfig(level=logging.ERROR)

### for large-scale test, shutdown visualization ###
# os.environ["MPLBACKEND"] = "Agg"  # must be before importing pyplot
# import matplotlib
# matplotlib.use("Agg")             # belt-and-suspenders
# import matplotlib.pyplot as plt
# plt.ioff()                        # avoid implicit interactive windows
### for large-scale test, shutdown visualization ###

robot = namedtuple('robot', 'shape radius length width min_speed max_speed max_acce kinematics')
obs   = namedtuple('obstacle', 'center radius vertex cone_type velocity')


def getRobotGoalTraj(target_traj, robot_pose, position, distance=2.0):
    if position not in position_dict:
        raise ValueError(
            f"Invalid position '{position}'. Expected one of: {list(position_dict.keys())}"
        )

    _diff = np.diff(target_traj, axis=0)
    target_yaws = np.atan2(_diff[:, 1], _diff[:, 0])
    target_yaws = np.append(target_yaws, target_yaws[-1])
    assert target_traj.shape[0] == target_yaws.shape[0]
    
    goal_traj = []
    for i in range(target_traj.shape[0]):
        target_person_pose = [target_traj[i, 0], target_traj[i, 1], target_yaws[i]]
        goal_traj.append(getRobotGoal(target_person_pose, robot_pose, position, distance))

    return goal_traj


def getRobotGoal(target_person_pose, robot_pose, position, distance=2.0):

    if position not in position_dict:
        raise ValueError(
            f"Invalid position '{position}'. Expected one of: {list(position_dict.keys())}"
        )

    target_person_pose = np.array(target_person_pose).squeeze()
    robot_pose = np.array(robot_pose).squeeze()

    if position == "back":
        # target pose for back following behavior
        alpha_back = np.atan2( target_person_pose[1] - robot_pose[1], 
                               target_person_pose[0] - robot_pose[0] )
        x = target_person_pose[0] - distance * np.cos(alpha_back)
        y = target_person_pose[1] - distance * np.sin(alpha_back)
        # The robot always faces the target person
        yaw = alpha_back

    elif position in ["left_side", "right_side"]:
        # target pose for left-side following behavior
        alpha_left = np.atan2( np.sin(position_dict["left_side"] + target_person_pose[2]), 
                               np.cos(position_dict["left_side"] + target_person_pose[2]) )
        x_left = target_person_pose[0] + distance * np.cos(alpha_left)
        y_left = target_person_pose[1] + distance * np.sin(alpha_left)
        dist_left = np.linalg.norm(robot_pose[:2] - [x_left, y_left])

        # target pose for right-side following behavior
        alpha_right = np.atan2( np.sin(position_dict["right_side"] + target_person_pose[2]), 
                                np.cos(position_dict["right_side"] + target_person_pose[2]) )
        x_right = target_person_pose[0] + distance * np.cos(alpha_right)
        y_right = target_person_pose[1] + distance * np.sin(alpha_right)
        dist_right = np.linalg.norm(robot_pose[:2] - [x_right, y_right])

        if position == "left_side":
            x, y = (x_left, y_left) if dist_left <= dist_right else (x_right, y_right)
        else:  # right_side
            x, y = (x_right, y_right) if dist_left >= dist_right else (x_left, y_left)
        
        # The robot's heading angle aligns with the target person
        yaw = target_person_pose[2]

    else:
        print(f"Error: Unknown position '{position}'. Expected values are 'back','left_side', or 'right_side'.")
 
    return np.array([x, y, yaw])[:, np.newaxis]


def getTargetPersonState(env, yaw_target_list):
    # omni: velocity = (vx, vy)     state = (x, y, yaw)
    target_person_vel = env.target.velocity
    target_person_pose = env.target.state

    # calculate the velocity and yaw of the target person
    v_target = np.sqrt(target_person_vel[0][0]**2 + target_person_vel[1][0]**2)
    yaw_target = np.atan2(target_person_vel[1][0], target_person_vel[0][0])
    if v_target < 0.01 and len(yaw_target_list) > 0:
        yaw_target = yaw_target_list[-1]
    yaw_target_list.append(yaw_target)

    if len(yaw_target_list) > 1:
        w_target = np.gradient(yaw_target_list, env.step_time)[-1]
    else:
        w_target = 0.0
        
    target_person_vel = np.array([[v_target], [w_target]])
    target_person_pose[2] = yaw_target

    return target_person_vel, target_person_pose


def scan_box(state, scan_data):

    ranges = np.array(scan_data['ranges'])
    angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

    point_list = []
    obstacle_list = []

    for i in range(len(ranges)):
        scan_range = ranges[i]
        angle = angles[i]

        if scan_range < ( scan_data['range_max'] - 0.01):
            point = np.array([ [scan_range * np.cos(angle)], [scan_range * np.sin(angle)]  ])
            point_list.append(point)

    if len(point_list) < 4:
        return obstacle_list

    else:
        point_array = np.hstack(point_list).T
        labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(point_array)

        for label in np.unique(labels):
            if label == -1:
                continue
            else:
                point_array2 = point_array[labels == label]
                rect = cv2.minAreaRect(point_array2.astype(np.float32))
                box = cv2.boxPoints(rect)
                local_center = np.array([[rect[0][0]], [rect[0][1]]])  # (2,1)

                vertices = box.T

                trans = state[0:2]
                rot = state[2, 0]
                R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                global_vertices = trans + R @ vertices
                global_center = (trans + R @ local_center).flatten()  # (2,)

                obstacle_list.append(obs(global_center, None, global_vertices, 'Rpositive', 0))

        return obstacle_list


def scan_points(state, scan_data):
    """
    Convert LIDAR scan results into obstacle points in the global frame

    args:
        state : np.ndarray
            Robot pose, expected shape (3,1) or (3,). Format: [x, y, theta]
        scan_data : dict
            LIDAR scan dictionary with keys:
                - 'ranges'    : iterable of range measurements (floats)
                - 'angle_min' : starting angle (radians)
                - 'angle_max' : ending angle (radians)
                - 'range_min' : minimum valid range (float)
                - 'range_max' : maximum valid range (float)

    returns:
        obstacles: np.ndarray 
            Obstacle points as an array of shape (N, 2), each row is [x, y] in global coordinates.
            If no valid points, returns an empty array with shape (0, 2).
    """

    ranges = np.array(scan_data['ranges'])
    angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))
    obstacles = []

    s = np.asarray(state).reshape(-1)
    if s.size < 3:
        raise ValueError("state must have at least 3 elements: [x, y, theta]")
    x, y, theta = float(s[0]), float(s[1]), float(s[2])

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    rmin = scan_data.get('range_min', 0.0)
    rmax = scan_data.get('range_max', np.inf)

    for r, a in zip(ranges, angles):
        if not np.isfinite(r):
            continue
        if r <= rmin or r >= (rmax - 0.01):
            continue
        local_point = np.array([r * np.cos(a), r * np.sin(a)])
        global_point = np.array([x, y]) + R @ local_point
        obstacles.append(global_point)

    if len(obstacles) == 0:
        return np.empty((0, 2))
        
    return np.vstack(obstacles)


def get_visible_humans(env):
    human_list = []
    for obj in env.visible_object_list:
        if obj.role == 'human':
            center = obj.state[0:2, 0]
            radius = obj.radius
            vertices = np.array([[center[0]-radius, center[1]-radius],
                                 [center[0]+radius, center[1]-radius],
                                 [center[0]+radius, center[1]+radius],
                                 [center[0]-radius, center[1]+radius]]).T
            # print(obj.state, obj.radius)
            human_list.append(obs(center, None, vertices, 'Rpositive', 0))
    return human_list


def main(world_name, args):

    if args.log_path != "":
        eval_dir = os.path.join(args.log_path, os.path.basename(world_name).replace('.yaml', ''))
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
    
    robot_info = env.get_robot_info()
    vel_min, vel_max = env.robot.get_vel_range()
    max_acce = robot_info.acce.flatten().tolist()
    robot_tuple = robot(robot_info.shape, env.robot.radius, env.robot.length, env.robot.width, 
                        vel_min.flatten().tolist(), vel_max.flatten().tolist(), max_acce, robot_info.kinematics)

    # DWA
    dwa_opt = DWA(robot_tuple, obstacle_radius=0.3, sample_time=env.step_time, predict_horizon=2.0, resolution_scale=(0.02, 0.02), 
                  weight_heading=3.0, weight_distance=2.0, weight_obstacle=1.0, weight_velocity=0.5)

    yaw_target_list = []

    # trajectory predictor
    traj_predictor_params = traj_predictors[args.traj_predictor]
    traj_predictor_params["dt"] = env.step_time  # Update the dt to match the environment's step time
    traj_predictor = get_predictor(traj_predictor_params["name"], traj_predictor_params)
    # x, y, vx, vy
    his_traj = {
        "updated_num": 0,
        "history_length": traj_predictor_params["history_length"],
        "dt": traj_predictor_params["dt"],
        "prediction_horizon":dwa_opt.predict_horizon,
        "robot": deque(maxlen=traj_predictor_params["history_length"]),
        "target": deque(maxlen=traj_predictor_params["history_length"]),
        "humans": {},
    }
    cg = curve_generator()

    if len(env.human_list) == 0:
        max_steps = args.min_steps
        max_search_steps = int(args.min_steps * 0.5 )
    else:
        max_steps = args.min_steps * 2
        max_search_steps = int(args.min_steps * 0.5 )

    for i in tqdm(range(max_steps)):

        target_person_vel, target_person_pose = getTargetPersonState(env, yaw_target_list)
        robot_vel  = env.robot.velocity  # [linear,angular]
        robot_pose = env.robot.state

        if env.check_target_visible:
            update_trajectory(env, his_traj)
            if his_traj["updated_num"] < traj_predictor_params["history_length"]:
                desired_pose = getRobotGoal(target_person_pose, robot_pose, args.position, args.distance+0.3)
                target_future_traj = None
                goal_traj = cg.generate_curve('dubins', [robot_pose, desired_pose], 0.1, 5)
                # print(len(goal_traj), goal_traj[0].shape)
            else:
                target_future_traj, _ = predict_trajectory(traj_predictor, his_traj)
                goal_traj = getRobotGoalTraj(target_future_traj, robot_pose, args.position, distance=args.distance+0.3)  # (3, N)
                if traj_predictor_params["name"] == "cvkf":
                    nis = target_future_traj[-1, 4]
                target_future_traj = target_future_traj[:, :2].T  # (2, N)
                # print(len(goal_traj), goal_traj[0].shape)
        # person search
        else:
            # print("Searching")
            target_future_traj, _ = predict_trajectory(traj_predictor, his_traj)
            predicted_target_pose = get_predicted_target_pose(target_future_traj)
            goal = getRobotGoal(predicted_target_pose, robot_pose, "back", distance=0.2)
            goal_traj = [goal for _ in range(target_future_traj.shape[0])]  # Create a constant goal trajectory
            target_future_traj = target_future_traj[:, :2].T  # (2, N)
            # print(len(goal_traj), goal_traj[0].shape)
        if enable_step_visuals:
            env.draw_trajectory(goal_traj, traj_type='k-', refresh=True)
            env.draw_points(goal_traj[0], s=60, c='r', refresh=True)

        scan_data = env.get_lidar_scan()
        obs_list = scan_points(robot_pose, scan_data)
        if obs_list.size == 0:
            masked_points = obs_list
        else:
            dists = np.linalg.norm(obs_list - target_person_pose[:2, 0], axis=1)
            masked_points = obs_list[dists > (env.target.radius + 0.2)]

        human_list = get_visible_humans(env)
        obs_dwa_list = masked_points.tolist()
        for human in human_list:
            obs_dwa_list.append(human.center)
        obs_dwa_list = np.array(obs_dwa_list)

        # # obstacles (center, radius, vertex, cone_type, velocity)
        # scan_data = env.get_lidar_scan()
        # obs_list = scan_box(robot_pose, scan_data)
        # human_list = get_visible_humans(env)
        # masked_obs_list = []
        # for obs in obs_list:
        #     dist = np.linalg.norm(obs.center - target_person_pose[:2, 0])
        #     if dist > env.target.radius+0.2:
        #         masked_obs_list.append(obs)
        # obs_list = masked_obs_list + human_list
        # for obs in obs_list:
        #     env.draw_box(obs.vertex, refresh=True)
        # # obs_list = []
        # obs_dwa_list = []
        # for obs in obs_list:
        #     obs_dwa_list.append(obs.center)
        # obs_dwa_list = np.array(obs_dwa_list)

        # DWA (ONLY work for DIFF robot)
        start_time = time.time()
        try:
            opt_vel, info = dwa_opt.control(robot_pose, robot_vel, goal_traj, obs_dwa_list)
        except Exception as e:
            print(f"Error in control: {e}")
            opt_vel = np.array([[0.0], [0.0]])
            info = {"arrive": False, "opt_state_list": [], "all_trajectories_list": []}
        alg_cost_t = time.time() - start_time

        # Draw a portion of candidate trajectories
        all_trajectories_num = 30
        all_trajectories = info.get('all_trajectories_list', [])
        if all_trajectories:
            if len(all_trajectories) > all_trajectories_num:
                indices = np.linspace(0, len(all_trajectories) - 1, all_trajectories_num, dtype=int)
                sampled_trajectories = [all_trajectories[i] for i in indices]
            else:
                sampled_trajectories = all_trajectories
            for traj in sampled_trajectories:
                env.draw_trajectory(traj, 'c', refresh=True)
        # Draw the optimal trajectory
        if enable_step_visuals:
            env.draw_trajectory(info['opt_state_list'], 'r', refresh=True)

        if args.visualize and not args.headless:
            env.save_figure(save_name="step_{:04d}.png".format(i))

        env.step(opt_vel)
        if enable_step_visuals:
            env.render(show_traj=True, show_trail=True)

        if eval_dir != "":
            env.record(alg_cost_t)

        if env.done():
            break
        
        # if info['arrive']:
        #     print('arrive at the goal')
        #     break

    if eval:
        print("evaluating...")
        env.eval(max_steps=max_steps, max_search_steps=max_search_steps)

    # env.end(ani_name='improve', show_traj=True, show_trail=True, ending_time=10,
    #         ani_kwargs={'subrectangles':True})


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_path", type=str, required=True, help="directory containing scenario yaml files")
    parser.add_argument("-s", "--scenario", type=str, default="1_dwa_test", help="scenario name")
    parser.add_argument("-p", "--position", type=str, default="left_side", help="back, left_side, right_side")
    parser.add_argument("-d", "--distance", type=float, default=1.3, help="distance to the target")
    parser.add_argument("-m", "--min_steps", type=int, default=1500, help="minimum steps required for target completing the whole trajectory without any human")

    parser.add_argument("-i", "--index", type=int, default=0, help="in this scenario, following position and distance, which index to run")
    parser.add_argument("-l", "--log_path", type=str, default="", help="evaluation log path")
    parser.add_argument("-t", "--traj_predictor", type=str, default="cvkf", help="trajectory predictor (cv, cvkf, sgan)")
    parser.add_argument("-v", "--visualize", action='store_true', default=False, help="whether to save per-step visualization frames")
    parser.add_argument("--headless", action='store_true', default=False, help="disable GUI display and per-step visualization for faster evaluation")

    args = parser.parse_args()

    if args.log_path == "":
        env_path_file = args.config_path + "/" + args.scenario + "_" + args.position + ".yaml"
    else:
        env_path_file = os.path.join(args.config_path, args.scenario + "_" + args.position + "_" + str(args.index) + ".yaml")

    main(env_path_file, args)
