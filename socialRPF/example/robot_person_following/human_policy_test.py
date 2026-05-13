import irsim
import numpy as np
import argparse
import cv2
import time

from RDA_planner.mpc_chasing_point2 import MPC
from collections import namedtuple
from sklearn.cluster import DBSCAN


position_dict = {
    "back": np.pi,
    "left_side": np.pi * 0.5,
    "right_side": np.pi * 1.5
}


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
        # In back-follow mode the robot always faces the target person; a 180-deg flip switches between front/back.
        angle_h2r = np.atan2(target_person_pose[1] - robot_pose[1], target_person_pose[0] - robot_pose[0])
        x = target_person_pose[0] - distance * np.cos(angle_h2r)
        y = target_person_pose[1] - distance * np.sin(angle_h2r)
        yaw = angle_h2r

    elif position in ["left_side", "right_side"]:
        # For left/right-side positions pick the closer candidate; the robot heading matches the target person's.
        alpha_l = position_dict["left_side"]
        normalized_alpha_l = np.atan2(np.sin(alpha_l + target_person_pose[2]), np.cos(alpha_l + target_person_pose[2]))
        x_l = target_person_pose[0] + distance * np.cos(normalized_alpha_l)
        y_l = target_person_pose[1] + distance * np.sin(normalized_alpha_l)
        dist_l = np.linalg.norm(robot_pose[:2] - [x_l, y_l])

        alpha_r = position_dict["right_side"]
        normalized_alpha_r = np.atan2(np.sin(alpha_r + target_person_pose[2]), np.cos(alpha_r + target_person_pose[2]))
        x_r = target_person_pose[0] + distance * np.cos(normalized_alpha_r)
        y_r = target_person_pose[1] + distance * np.sin(normalized_alpha_r)
        dist_r = np.linalg.norm(robot_pose[:2] - [x_r, y_r])

        # Pick the goal point according to the requested side preference
        if position == "left_side":
            x, y = (x_l, y_l) if dist_l <= dist_r else (x_r, y_r)
        else:  # right_side
            x, y = (x_r, y_r) if dist_r <= dist_l else (x_l, y_l)

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
    

def main(world_name, position, distance, max_steps, eval, headless=False):
    
    enable_display = not headless
    enable_step_visuals = not headless

    env = irsim.make(world_name=world_name, save_ani=False, display=enable_display, full=False, eval=eval)
    
    robot_info = env.get_robot_info()
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.wheelbase, 
                    [10, 1], [10, 0.5], 'diff')
    
    mpc_opt = MPC(car_tuple, None, receding=20, sample_time=env.step_time,  process_num=4, iter_num=4, 
                  max_edge_num=4, max_obs_num=3, obstacle_order=True, ws=10.0, wu=2.0, slack_gain=10)
    
    yaw_target_list = []

    for i in range(max_steps):
        # print(i)

        # print("G:", robot_info.G)
        # print("h:", robot_info.h)
        
        target_person_vel, target_person_pose = getTargetPersonState(env, yaw_target_list)
        # print("target person velocity:", target_person_vel, "\ntarget person pose:", target_person_pose)

        robot_pose = env.robot.state
        # print("robot pose: ", robot_pose)

        goal = getRobotGoal(target_person_pose, robot_pose, position, distance=distance)
        # print("goal position:", goal)
        if enable_step_visuals:
            env.draw_points(goal[0:2], s=60, c='r', refresh=True)

        # obstacles (center, radius, vertex, cone_type, velocity)
        scan_data = env.get_lidar_scan()
        obs_list = scan_box(robot_pose, scan_data)
        if enable_step_visuals:
            for obs in obs_list:
                env.draw_box(obs.vertex, refresh=True)
        # obs_list = []  # not consider obstacle avoidance
        
        start_time = time.time()
        opt_vel, info = mpc_opt.control(robot_pose, goal, target_person_vel, 4, obs_list, follow_type="back")
        
        # print("visible:", env.visible_object_list)

        # predicting trajectory of the mpc
        if enable_step_visuals:
            env.draw_trajectory(info['opt_state_list'], 'r', refresh=True)
        # print("MPC: {}".format(time.time()-start_time))
        # print("opt_vel: {}".format(opt_vel))
        
        env.step(opt_vel)
        if enable_step_visuals:
            env.render(show_traj=True, show_trail=True)

        if eval:
            env.record()

        if env.done():
            break
        
        # if info['arrive']:
            # print('arrive at the goal')
            # break

    if eval:
        print("evaluating...")
        env.eval()

    env.end(ani_name='rda_planner_following', show_traj=True, show_trail=True, ending_time=10, 
            ani_kwargs={'subrectangles':True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario", type=str, default="inverseFlow", help="scenario")
    parser.add_argument("-p", "--position", type=str, default="back", help="back, left_side, right_side")
    parser.add_argument("-d", "--distance", type=float, default=2.0, help="distance to the target")
    parser.add_argument("-e", "--eval", type=bool, default=False, help="whether to record data for evaluation")
    parser.add_argument("-m", "--max_steps", type=int, default=500, help="simulated steps")
    parser.add_argument("--headless", action='store_true', default=False, help="disable GUI display and per-step visualization for faster evaluation")

    args = parser.parse_args()

    env_path_file = "config/" + args.scenario + "_" + args.position + ".yaml"

    main(env_path_file, args.position, args.distance, args.max_steps, args.eval, args.headless)
