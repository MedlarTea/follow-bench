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

# from DWA_planner.dwa import DWA
from SFM_planner.sfm import social_force_model
from collections import namedtuple
from sklearn.cluster import DBSCAN

from irsim.util.util import relative_position, WrapToPi, omni_to_diff

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

car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce dynamics')
obs = namedtuple('obstacle', 'center radius vertex cone_type velocity')


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
                center = np.array(rect[0])

                vertices = box.T

                trans = state[0:2]
                rot = state[2, 0]
                R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                global_vertices = trans + R @ vertices
                global_center = trans + R @ center.reshape(2,1)

                obstacle_list.append(obs(np.squeeze(global_center), None, global_vertices, 'Rpositive', 0))

        return obstacle_list

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
            human_list.append(obs(center, radius, vertices, 'Rpositive', 0))
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
        save_ani=False,
        display=enable_display,
        full=False,
        eval_dir=eval_dir,
    )
    
    robot_info = env.get_robot_info()
    # car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.wheelbase, 
                    # [2.0, np.pi], [2.0, 1.0*np.pi], 'diff')
   
    # DWA
    sfm_params = {
      "vxmax": 3.0,         # max linear velocity along x (m/s)
      "vymax": 3.0,         # max linear velocity along y (m/s)
      "acceler": 3.0,       # max resultant acceleration (m/s^2)

      # SFM force coefficients --------------------------
      "gain_k": 1.3,        # goal attraction gain k
      "gain_a_static": 10,  # static-obstacle repulsion strength a
      "gain_b_static": 0.5, # static-obstacle repulsion decay b
      "gain_a_agent": 7,    # pedestrian-pedestrian repulsion strength a
      "gain_b_agent": 0.5,  # pedestrian-pedestrian repulsion decay b

      # Distance thresholds ----------------------------
      "avoid_dis": 5.0,     # interaction radius (m): repulsion is ignored beyond this
      "safe_dist": 0.3,     # minimum safe distance (m)
    }
    # print(robot_info)
    # print(robot_info.G)
    robot_radius = np.max(np.abs(robot_info.G))/2
    sfm_opt = social_force_model(radius=robot_radius, sample_time=env.step_time, predict_horizon=2.0, **sfm_params)

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
        "robot": deque(maxlen=traj_predictor_params["history_length"]),
        "target": deque(maxlen=traj_predictor_params["history_length"]),
        "humans": {},
    }

    if len(env.human_list) == 0:
        max_steps = args.min_steps
        max_search_steps = int(args.min_steps * 0.5 )
    else:
        max_steps = args.min_steps * 2
        max_search_steps = int(args.min_steps * 0.5 )

    for i in tqdm(range(max_steps)):
        
        target_person_vel, target_person_pose = getTargetPersonState(env, yaw_target_list)
        # print("target person velocity:", target_person_vel, "\ntarget person pose:", target_person_pose)
        # print("robot radius:", robot_radius)
        # print("target person velocity:", target_person_vel)
        
        robot_vel  = env.robot.velocity_xy  # [vx, vy]
        robot_pose = env.robot.state
        # print("robot velocity:", robot_vel, "\nrobot pose: ", robot_pose)
        # print("robot velocity:", robot_vel)

        # person following
        if env.check_target_visible:
            update_trajectory(env, his_traj)
            goal = getRobotGoal(target_person_pose, robot_pose, args.position, distance=args.distance)
        # person search
        else:
            print("Searching")
            target_future_traj, _ = predict_trajectory(traj_predictor, his_traj)
            preeicted_target_pose = get_predicted_target_pose(target_future_traj)
            goal = getRobotGoal(preeicted_target_pose, robot_pose, "back", distance=0.2)
    
        # print("goal position:", goal)
        if enable_step_visuals:
            env.draw_points(goal[0:2], s=60, c='r', refresh=True)
        # print("relative distance: ", np.sqrt( (goal[0]-robot_pose[0])**2 + (goal[1]-robot_pose[1])**2 ))

        # obstacles (center, radius, vertex, cone_type, velocity)
        scan_data = env.get_lidar_scan()
        obs_list = scan_box(robot_pose, scan_data)
        human_list = get_visible_humans(env)
        masked_obs_list = []
        for obs in obs_list:
            dist = np.linalg.norm(obs.center - target_person_pose[:2, 0])
            if dist > env.target.radius+0.2:
                masked_obs_list.append(obs)
        obs_list = masked_obs_list
        if enable_step_visuals:
            for obs in obs_list:
                env.draw_box(obs.vertex, refresh=True)
        
        static_obs_sfm_list = []
        for obs in masked_obs_list:
            for human in human_list:
                dist = np.linalg.norm(obs.center - human.center)
                if dist < env.target.radius+0.2:
                    continue
            static_obs_sfm_list.append(obs.center)
        human_sfm_list = []
        for human in human_list:
            human_sfm_list.append(human.center)
            
        # DWA (ONLY work for OMNI robot)
        start_time = time.time()
        try:
            opt_vel, info = sfm_opt.control(robot_pose, robot_vel, goal, static_obs_sfm_list, human_sfm_list)
        except Exception as e:
            print(f"Error in SFM control: {e}")
            opt_vel = np.array([[0.0], [0.0]])
            info = {"arrive": False, "opt_state_list": []}

        alg_cost_t = time.time() - start_time

        # print(alg_cost_t)

        # print("yaw:", robot_pose[2, 0])
        # print("desired yaw:", goal[2, 0])
        # print("opt_vel:", opt_vel)
        # v = np.sqrt(opt_vel[0, 0]**2 + opt_vel[1, 0]**2)
        # w = WrapToPi(goal[2, 0] - robot_pose[2, 0]) / env.step_time
        # opt_vel = np.array([[v], [w]])
        opt_vel = omni_to_diff(robot_pose[2, 0], opt_vel, w_max=2.0)
        # print("after opt_vel:", opt_vel)
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
        
        if info['arrive']:
            print('arrive at the goal')
            break

    if eval_dir != "":
        print("evaluating...")
        env.eval(max_steps=max_steps, max_search_steps=max_search_steps)

    # env.end(ani_name='dwa_planner_following', show_traj=True, show_trail=True, ending_time=10, 
            # ani_kwargs={'subrectangles':True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="directory containing scenario yaml files")
    parser.add_argument("-s", "--scenario", type=str, default="square", help="scenario name")
    parser.add_argument("-p", "--position", type=str, default="back", help="back, left_side, right_side")
    parser.add_argument("-d", "--distance", type=float, default=1.5, help="distance to the target")
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
