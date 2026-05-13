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

from RDA_planner.mpc_chasing_point2 import MPC
from collections import namedtuple
from sklearn.cluster import DBSCAN
from traj_predictor import get_predictor
from collections import deque
from global_params import traj_predictors, position_dict, predict_trajectory, update_trajectory, get_predicted_target_pose, generateSamplePoints_korobov_lattice, get_yaw

from global_params import gamma, sigma

from shapely.geometry import Polygon, Point
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

    if position == "front":
        yaw = target_person_pose[2]
        a = yaw + position_dict["front"] + 2 * np.pi if yaw < 0 else yaw + position_dict["front"]  # add in [0, 2pi]
        x = target_person_pose[0] + np.cos(a) * distance
        y = target_person_pose[1] + np.sin(a) * distance

    elif position == "back":
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
            human_list.append(obs(None, None, vertices, 'Rpositive', 0))
    return human_list

def find_occluder(env, robot_position, last_target_position, observed_human_ids):
    closest_dist = float('inf')
    closest_occluder_id = None
    # closest_occluder_velocity = None

    # print(robot_position)
    # print(last_target_position)

    # Build the robot-target line
    line_vec = last_target_position - robot_position
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        return None  # robot and target overlap

    line_dir = line_vec / line_length  # unit direction vector

    for obj in env.visible_object_list:
        if obj.role == 'human' and obj.id in observed_human_ids:
            # Center coordinate of the human (obj.state is 3x1 or 4x4, first two dims are x, y)
            center = obj.state[0:2, 0]  # state as 2D coordinate matrix
            # velocity = obj.velocity_xy[:2, 0]  # human velocity

            # Project the human onto the robot-target segment
            vec_to_center = center - robot_position
            proj_length = np.dot(vec_to_center, line_dir)

            # Restrict the projection to [0, line_length]
            if proj_length < 0 or proj_length > line_length:
                continue

            # Perpendicular distance to the segment
            proj_point = robot_position + proj_length * line_dir
            dist_to_line = np.linalg.norm(center - proj_point)

            # Pick the human closest to the line
            if dist_to_line < closest_dist:
                closest_dist = dist_to_line
                closest_occluder_id = obj.id
                # closest_occluder_velocity = velocity

    return closest_occluder_id

def get_triangle(sample_point, target_point, target_radius):
    p = np.array(sample_point, dtype=float)
    c = np.array(target_point, dtype=float)
    r = target_radius

    d = c - p
    D = np.linalg.norm(d)

    if D <= r:
        # raise ValueError("Sample point is inside or on the circle; no external tangents.")
        return None

    d_hat = d / D
    n_hat = np.array([-d_hat[1], d_hat[0]])  # normal vector

    theta = np.arcsin(r / D)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    v1 = cos_theta * d_hat + sin_theta * n_hat
    v2 = cos_theta * d_hat - sin_theta * n_hat

    t1 = c - r * np.array([-v1[1], v1[0]])
    t2 = c + r * np.array([-v2[1], v2[0]])

    return Polygon([p.tolist(), t1.tolist(), t2.tolist()])

def get_cylinder_bottom_center(last_point, current_point, bias=2.0):
    direction = current_point - last_point
    angle = np.arctan2(direction[1], direction[0])
    cylinder_bottom_center = np.array([current_point[0] + bias*np.cos(angle), current_point[1] + bias*np.sin(angle), 0.0])
    return cylinder_bottom_center

def cal_dist_to_closest_point(point, positions):
    point = np.array(point)
    positions = np.array(positions)
    distances = np.linalg.norm(positions - point, axis=1)
    closest_distance = np.min(distances)
    return closest_distance

def cal_cost_distanceToPerson(sample_point, predict_distracter_positions, max_dist=3.0):
    """
    Lower cost is better.
    """
    dist_ToPerson = cal_dist_to_closest_point(sample_point, predict_distracter_positions)
    if dist_ToPerson > max_dist:
        return 0.0
    cost_ToPerson = (1/dist_ToPerson - 1/max_dist) / dist_ToPerson**2
    return cost_ToPerson

def select_overtake_pose(env, robot_pose, fake_target_pose, historical_occluder_positions, predicted_occluder_positions, n_sampled_point=10, iou_threshold=0.3, cost_threshold=3.0):
    min_cost = 1000
    sample_points = []
    best_sample_point = None
    robot_position = robot_pose[:2, 0]
    occluder_id = find_occluder(env, robot_position, fake_target_pose, historical_occluder_positions.keys())

    if occluder_id not in predicted_occluder_positions:
        print(f"Occluder ID {occluder_id} not found in predicted occluder positions.")
        return sample_points, occluder_id, best_sample_point, min_cost

    predicted_occluder_positions = predicted_occluder_positions[occluder_id][:, :2] if occluder_id is not None else []

    historical_occluder_positions = np.array(historical_occluder_positions[occluder_id])[:, :2] if occluder_id is not None else []

    sample_points = generateSamplePoints_korobov_lattice(robot_position, historical_occluder_positions[-1], in_radius=0.5, out_radius=3.0, num_points=n_sampled_point)

    local_map = env.robot.lidar.local_map

    for i, sample_point in enumerate(sample_points):
        # 1. check sampled point with local map
        ego_pose = robot_pose[:3, 0]
        cos_theta = np.cos(ego_pose[2])
        sin_theta = np.sin(ego_pose[2])
        dx = sample_point[0] - ego_pose[0] 
        dy = sample_point[1] - ego_pose[1]
        x_ego = cos_theta * dx + sin_theta * dy
        y_ego = -sin_theta * dx + cos_theta * dy
        rel_position = np.array([x_ego, y_ego])
        x = int(round(rel_position[0]/local_map.resolution+local_map.x_center))
        y = int(round(rel_position[1]/local_map.resolution+local_map.x_center))
        # out of perception range
        if x < 0 or x >= local_map.map.shape[1] or y < 0 or y >= local_map.map.shape[0]:
            # print("OUT OF RANGE")
            continue
        # occupied
        if local_map.map[y, x] == 1.0:
            # print("Occupied")
            continue

        # 2. check sampled point with predicted occluder's positions for occlusion avoidance
        cost_to_occluder = cal_cost_distanceToPerson(sample_point, predicted_occluder_positions, max_dist=2.0)

        # 3. check sampled point with target person for better observation
        yaw = get_yaw(sample_point, fake_target_pose[:2])

        # triangle
        triangle_to_target = get_triangle(sample_point, fake_target_pose[:2], env.target.radius)
        triangle_to_occluder = get_triangle(sample_point, predicted_occluder_positions[0], env.target.radius)
        if triangle_to_target is None or triangle_to_occluder is None:
            continue

        intersection = triangle_to_target.intersection(triangle_to_occluder)
        union = triangle_to_target.union(triangle_to_occluder)
        cost_iou = intersection.area / union.area if not intersection.is_empty else 0.0

        if cost_iou > iou_threshold:
            # print("Overlapped")
            continue  # skip this sample point if IoU is too high

        # 4. calculate the dist
        cost_attractive =  np.linalg.norm(sample_point - robot_position) / (1 - cost_iou)
    
        cost_all = cost_to_occluder + cost_attractive
        if cost_all > cost_threshold:
            # print("Cost too high")
            continue

        if cost_all < min_cost:
            min_cost = cost_all
            best_sample_point = np.array([sample_point[0], sample_point[1], yaw])

    return sample_points, occluder_id, best_sample_point, min_cost

def get_velocity(env, x, gamma):
    alpha_sum = 0
    vx_sum = 0
    vy_sum = 0
    for obj in env.visible_object_list:
        if obj.role == 'human':
            distance_sq = np.linalg.norm(x - obj.state[:2, 0]) ** 2
            alpha = np.exp(-gamma * distance_sq)
            vx_sum += alpha * obj.velocity_xy[0, 0]
            vy_sum += alpha * obj.velocity_xy[1, 0]
            alpha_sum += alpha
    if alpha_sum > 0:
        vx_sum = vx_sum / alpha_sum
        vy_sum = vy_sum / alpha_sum
    return np.array([vx_sum, vy_sum])

def get_density(env, x, sigma):
    density = 0
    N_obs = 1  # Assume each position has been observed once
    for obj in env.visible_object_list:
        if obj.role == 'human':
            distance_sq = np.linalg.norm(x - obj.state[:2, 0]) ** 2
            density += np.exp(-distance_sq / (2 * sigma ** 2))
    rho = (1 / (2 * np.pi * sigma ** 2)) * (1 / N_obs) * density
    return rho

def compute_objective_function(env, sigma, gamma, search_direction, n_sampled_point=30):
    P_robot = env.robot.state[:2, 0]
    V_robot = env.robot.velocity_xy[:2, 0]

    search_direction = search_direction / np.linalg.norm(search_direction)  # Normalize the search direction

    sample_points = generateSamplePoints_korobov_lattice(P_robot, P_robot+0.5*search_direction, in_radius=0.0, out_radius=2.5, num_points=n_sampled_point)
    # Compute objective function for sample points
    O_sample_points = np.zeros(len(sample_points))
    for i, point in enumerate(sample_points):
        v_field = get_velocity(env, point, gamma)
        velocity_diff = np.linalg.norm(V_robot - v_field)
        position_diff = np.linalg.norm(P_robot - point)
        O_sample_points[i] = max(get_density(env, point, sigma),0.1) * gamma * max(velocity_diff,0.1) * max(1.5, position_diff)

    
    return sample_points, O_sample_points

def select_fluidfollowing_pose(env, robot_pose, sigma, gamma, search_direction, n_sampled_point=30):
    robot_position = robot_pose[:2, 0]
    # Compute the objective function
    sample_points, O_sample_points = compute_objective_function(env, sigma, gamma, search_direction, n_sampled_point)

    # Find the optimal point with minimum cost
    optimal_idx = np.argmin(O_sample_points)
    optimal_x, optimal_y = sample_points[optimal_idx]

    return sample_points, np.array([optimal_x, optimal_y, get_yaw(robot_position, [optimal_x, optimal_y])])

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
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.wheelbase, 
                    [robot_info.vel_max[0,0], robot_info.vel_max[1,0]], [robot_info.acce[0, 0], robot_info.acce[1, 0]], 'diff')
    
    mpc_opt = MPC(car_tuple, None, receding=10, sample_time=env.step_time, process_num=4, iter_num=4, 
                  max_edge_num=4, max_obs_num=5, obstacle_order=True, ws=10.0, wu=1.0, slack_gain=10)
    
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
    
    INITIALIZED_SEARCH_DIRECTION = False

    for i in tqdm(range(max_steps)):
        target_person_vel, target_person_pose = getTargetPersonState(env, yaw_target_list)
        robot_pose = env.robot.state

        # person following
        if env.check_target_visible:
            update_trajectory(env, his_traj)
            goal = getRobotGoal(target_person_pose, robot_pose, args.position, distance=args.distance)
            INITIALIZED_SEARCH_DIRECTION = False
        # person search
        else:
            print("Searching")
            target_future_traj, predicted_human_traj = predict_trajectory(traj_predictor, his_traj)

            if not INITIALIZED_SEARCH_DIRECTION:
                # Initialize search direction based on the last known target position
                search_direction = np.array(his_traj["target"][-1][:2]) - np.array(robot_pose[:2, 0])
                INITIALIZED_SEARCH_DIRECTION = True

            fake_target_pose = robot_pose[:2, 0] + search_direction
            sample_points, occluder_id, best_overtake_pose, min_cost = select_overtake_pose(env, robot_pose, fake_target_pose, his_traj["humans"], predicted_human_traj, n_sampled_point=20, iou_threshold=0.3, cost_threshold=3.5)
            if enable_step_visuals:
                env.draw_points(sample_points, s=40, c='black', refresh=True)

            # overtaking
            if best_overtake_pose is not None:
            # if False:
                goal = best_overtake_pose[:, np.newaxis]
            # fluid following
            else:
                sample_points, best_fluid_following_pose = select_fluidfollowing_pose(env, robot_pose, sigma, gamma, search_direction, n_sampled_point=30)
                goal = best_fluid_following_pose[:, np.newaxis]
                if enable_step_visuals:
                    env.draw_points(sample_points, s=40, c='black', refresh=True)
        
        if enable_step_visuals:
            env.draw_points(goal[0:2], s=60, c='r', refresh=True)

        # obstacles (center, radius, vertex, cone_type, velocity)
        scan_data = env.get_lidar_scan()
        obs_list = scan_box(robot_pose, scan_data)
        human_list = get_visible_humans(env)
        masked_obs_list = []
        for obs in obs_list:
            dist = np.linalg.norm(obs.center - target_person_pose[:2, 0])
            if dist > env.target.radius+0.2:
                masked_obs_list.append(obs)
        obs_list = masked_obs_list + human_list
        if enable_step_visuals:
            for obs in obs_list:
                env.draw_box(obs.vertex, refresh=True)

        start_time = time.time()
        try:
            opt_vel, info = mpc_opt.control(robot_pose, goal, 4, obs_list)
        except Exception as e:
            print(f"Error in control: {e}")
            opt_vel = np.array([[0.0], [0.0]])
            info = {"arrive": False, "opt_state_list": []}
        alg_cost_t = time.time() - start_time
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

    if eval_dir != "":
        print("evaluating...")
        env.eval(max_steps=max_steps, max_search_steps=max_search_steps)

    # env.end(ani_name='rda_planner_following', show_traj=True, show_trail=True, ending_time=10, 
    #         ani_kwargs={'subrectangles':True})


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
