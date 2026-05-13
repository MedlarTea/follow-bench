import os
import json
import numpy as np
import matplotlib.pyplot as plt
def eval(max_steps=400, max_search_steps=100, eval_dir=None, zone_thresholds = [0.45, 1.2]):
        if eval_dir is not None:
            eval_dir = eval_dir
            object_fname = os.path.join(eval_dir, 'object_info.json')
            basic_info_fname = os.path.join(eval_dir, 'env_basic_info.json')
            grid_map_fname = os.path.join(eval_dir, 'grid_map.png')
        
        # load object info
        # with open(object_fname, 'r') as f:
        #     object_info = json.load(f)
        
        # load basic info
        with open(basic_info_fname, 'r') as f:
            basic_info = json.load(f)

        # load grid map
        if os.path.exists(grid_map_fname):
            grid_map = plt.imread(grid_map_fname)

        # env basic info
        step_time = basic_info['step_time']
        robot_radius = basic_info['robot']['radius']
        robot_length = basic_info['robot']['length']
        robot_width = basic_info['robot']['width']

        target_radius = basic_info['target']['radius']

        human_basic_info = basic_info['human']

        # evaluated metrics
        obstacle_avoidance_success = True
        search_success = True
        robot_target_dists = []
        robot_positions = []
        robot_velocities = []
        target_positions = []
        time_in_personal_zone = 0
        time_in_private_zone = 0
        counts = 0

        target_missed = []
        time_in_search = 0
        
        target_missing_counts = 0

        # object info
        with open(object_fname, encoding="utf-8") as f:
            for line in f:
                object_info = json.loads(line)
                counts = object_info['count']
                target_visible = object_info['target_visible']
                robot_info = object_info['robot'][0]
                target_info = object_info['target'][0]
                obstacle_info = object_info['obstacles']
                human_info = object_info['humans']

                if target_visible == 0:
                    time_in_search += step_time
                    target_missed.append(1)
                    target_missing_counts += 1
                    if target_missing_counts > max_search_steps:
                        search_success = False
                else:
                    target_missed.append(0)
                    target_missing_counts = 0

                robot_position = robot_info[1:3]  # x, y
                robot_positions.append(robot_position)
                robot_velocities.append(robot_info[4:6])
                if robot_info[6]:
                    obstacle_avoidance_success = False

                target_position = target_info[1:3]
                target_positions.append(target_position)
                robot_target_dist = np.linalg.norm(np.array(robot_position) - np.array(target_position))
                robot_target_dists.append(robot_target_dist.tolist())

                robot_target_dist_with_radius = robot_target_dist - (robot_radius + target_radius)

                if robot_target_dist_with_radius > zone_thresholds[0] and robot_target_dist_with_radius < zone_thresholds[1]:
                    time_in_personal_zone += step_time

                closest_human_dist = min([np.linalg.norm(np.array(robot_position) - np.array(human[1:3])) for human in human_info]) if len(human_info) > 0 else 100

                robot_human_dist_with_radius = closest_human_dist - (robot_radius + target_radius)


                if robot_human_dist_with_radius < zone_thresholds[0]:
                    time_in_private_zone += step_time

        robot_positions = np.array(robot_positions)
        robot_velocities = np.array(robot_velocities)
        robot_target_dists = np.array(robot_target_dists)
        target_positions = np.array(target_positions)
        target_missed = np.array(target_missed)

        print(target_missed.shape)

        # print("robot_position shape:", robot_positions.shape)
        # print(robot_positions[:20, :])
        robot_path_length = np.diff(robot_positions, axis=0)
        # print("robot_path_length shape:", robot_path_length.shape)
        # print(robot_path_length[:20, :])
        robot_path_length = np.linalg.norm(robot_path_length, axis=1)
        # print("robot_path_length norm shape:", robot_path_length.shape)
        # print(robot_path_length[:20])
        total_path_length = robot_path_length.sum().tolist()
        # print(total_path_length)

        robot_search_length = robot_path_length[target_missed[1:] == 1].sum().tolist()

        target_visibility_ratio = sum(target_missed == 0) / max_steps

        robot_accelerations = np.diff(robot_velocities, axis=0, prepend=robot_velocities[:1]) / step_time
        robot_accelerations = np.linalg.norm(robot_accelerations, axis=1)


        robot_jerk = np.diff(robot_velocities, n=2, axis=0, prepend=robot_velocities[:2]) / step_time**2
        robot_jerk = np.linalg.norm(robot_jerk, axis=1)

        # save eval result
        # position_fig = os.path.join(eval_dir, 'position.png')
        # draw_traj(robot_positions, target_positions, position_fig)

        # robot_target_dist_fig = os.path.join(eval_dir, 'robot_target_dist.png')
        # draw_motion_info(robot_target_dists, "Robot Target Distance", "Distance [m]", robot_target_dist_fig)

        # robot_velocities = np.linalg.norm(robot_velocities, axis=1)
        # velocity_fig = os.path.join(eval_dir, 'velocity.png')
        # draw_motion_info(robot_velocities, "Robot Velocity", "Velocity [m/s]", velocity_fig)

        # acceleration_fig = os.path.join(eval_dir, 'acceleration.png')
        # draw_motion_info(robot_accelerations, "Robot Acceleration", "Acceleration [m/s^2]", acceleration_fig)

        # jerk_fig = os.path.join(eval_dir, 'jerk.png')
        # draw_motion_info(robot_jerk, "Robot Jerk", "Jerk [m/s^3]", jerk_fig)

        eval_result_fname = os.path.join(eval_dir, 'eval_result_new.json')
        eval_result = {
            "max_steps": max_steps,
            "max_search_steps": max_search_steps,
            "obstacle_avoidance_success": obstacle_avoidance_success,
            "target_visibility_ratio": target_visibility_ratio,
            "search_success": search_success,
            "search_path_length": robot_search_length,

            "avg_robot_target_dist_no_radius": np.mean(robot_target_dists).tolist(),
            "path_length": total_path_length,
            "avg_velocity": np.mean(robot_velocities).tolist(),
            "avg_acceleration": np.mean(robot_accelerations).tolist(),
            "avg_jerk": np.mean(robot_jerk).tolist(),
            "time_in_target_personal_zone": time_in_personal_zone,
            "time_in_human_private_zone": time_in_private_zone,
            "time_in_target_search": time_in_search,
            "total_time": counts*step_time,
        }
        with open(eval_result_fname, "w") as f:
            json.dump(eval_result, f, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-run the evaluation pipeline on a single trial directory "
                    "and write eval_result_new.json next to the source files."
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Path to a single trial directory, e.g. "
             "<log_root>/<planner>/<scenario>_<position>_<index>/",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-search-steps", type=int, default=125)
    parser.add_argument(
        "--zone-thresholds",
        type=float,
        nargs=2,
        default=[0.45, 1.2],
        metavar=("INNER", "OUTER"),
        help="Personal-zone thresholds (in metres).",
    )
    args = parser.parse_args()

    eval(
        max_steps=args.max_steps,
        max_search_steps=args.max_search_steps,
        eval_dir=args.eval_dir,
        zone_thresholds=list(args.zone_thresholds),
    )