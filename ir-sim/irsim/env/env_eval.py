import matplotlib.pyplot as plt
from irsim.global_param.path_param import path_manager as pm
from irsim.global_param import world_param, env_param
from math import sin, cos
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import shutil
class EnvEval:
    """
    EnvPlot class for visualizing the environment.

    Args:
        grid_map (optional): The grid map of the environment (PNG file).
        objects: List of objects in the environment.
        x_range (list): The range of x-axis values. Default is [0, 10].
        y_range (list): The range of y-axis values. Default is [0, 10].
        saved_figure (dict): Keyword arguments for saving the figure.
            See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html for details.
        saved_ani (dict): Keyword arguments for saving the animation.
            See https://imageio.readthedocs.io/en/v2.8.0/format_gif-pil.html#gif-pil for details.
        dpi: Dots per inch for the figure. Default is 100.
        figure_pixels: Width and height of the figure in pixels. Default is [1920, 1080].
        kwargs: Additional options such as color_map, no_axis, and tight.
    """

    def __init__(
        self,
        eval_dir="",
        grid_map=None,
        objects=[],
        **kwargs,
    ) -> None:
        """
        Initialize the EnvPlot instance.
        """
        self.eval_dir = eval_dir

        self.init_eval_log(self.eval_dir, grid_map, objects)

        self.count = 1

    def init_eval_log(self, eval_dir, grid_map, objects):
        """eval config
        
        """
        # check config
        if eval_dir == "":
            return       

        # make dir
        os.makedirs(self.eval_dir, exist_ok=True)

        # object info fname
        self.object_fname = os.path.join(self.eval_dir, 'object_info.json')
        self.object_buffer_fname = os.path.join(self.eval_dir, 'object_info_buffer.json')
        # if config["init_recorder"]:
        # open(self.object_fname, "w").close()  # empty the file


        # save grid map
        self.grid_map_fname = os.path.join(self.eval_dir, 'grid_map.png')
        if grid_map is not None:
            plt.imsave(self.grid_map_fname, grid_map)

        # save basic info
        self.basic_info_fname = os.path.join(self.eval_dir, 'env_basic_info.json')
        record_info = {
            'control_mode': world_param.control_mode,
            'collision_mode': world_param.collision_mode,
            'step_time': world_param.step_time,
            'robot': {},
            'obstacles': {},
            'human': {},
            'target': {},
        }
        for obj in objects:
            if obj.role == "robot":
                record_info['robot']["radius"] = obj.radius
                record_info['robot']["length"] = obj.length
                record_info['robot']["width"] = obj.width
            elif obj.role == "target":
                record_info['target']["radius"] = obj.radius
            elif obj.role == "obstacle":
                record_info['obstacles'][obj.id] = {"radius": obj.radius}
            elif obj.role == "human":
                record_info['human'][obj.id] = {"radius": obj.radius}

        with open(self.basic_info_fname, 'w') as f:
            json.dump(record_info, f, indent=4)

    def record(self, objects = [], alg_cost_t=0.0):
        if self.count == 1:
            # if the first record, create the file
            if os.path.exists(self.object_buffer_fname):
                os.remove(self.object_buffer_fname)

        record_info = {
            "count": self.count,
            'target_visible': 0,
            'alg_cost_t': alg_cost_t,
            'robot': [],
            'obstacles': [],
            'humans': [],
            'target': [],
        }
        # print("Count: {}, Alg Cost Time: {:.4f}s".format(world_param.count, alg_cost_t))

        for obj in objects:
            if obj.role == "robot":
                record_info['robot'].append([obj.id] + obj.state.squeeze().tolist() + obj.velocity_xy.squeeze().tolist() + [obj.collision])
                visible_objs = obj.get_visible_objects()
            elif obj.role == "target":
                record_info['target'].append([obj.id] + obj.state.squeeze().tolist() + obj.velocity_xy.squeeze().tolist())
            elif obj.role == "obstacle":
                record_info['obstacles'].append([obj.id] + obj.state.squeeze().tolist())
            elif obj.role == "human":
                record_info['humans'].append([obj.id] + obj.state.squeeze().tolist() + obj.velocity_xy.squeeze().tolist())
        

        for vis_obj in visible_objs:
            if vis_obj.role == "target":
                record_info['target_visible'] = 1
                break
        # robot_info = record_info['robot'][0]
        # robot_position = robot_info[1:3]
        # target_info = record_info['target'][0]
        # target_position = target_info[1:3]
        # robot_target_dist = np.linalg.norm(np.array(robot_position) - np.array(target_position))
        # robot_radius = 0.3
        # target_radius = 0.45
        # robot_target_dist_with_radius = robot_target_dist - (robot_radius + target_radius)
        # print("[DEBUG] rt_dist: {:.3f}".format(robot_target_dist_with_radius, ))
            
        with open(self.object_buffer_fname, mode="a", encoding="utf-8") as f:
            json.dump(record_info, f, ensure_ascii=False)
            f.write("\n")          # 每条记录占一行
        self.count += 1

    def eval(self, max_steps=400, max_search_steps=100, eval_dir=None, zone_thresholds = [0.45, 1.2]):
        if eval_dir is not None:
            self.eval_dir = eval_dir
            self.object_fname = os.path.join(self.eval_dir, 'object_info.json')
            self.basic_info_fname = os.path.join(self.eval_dir, 'env_basic_info.json')
            self.grid_map_fname = os.path.join(self.eval_dir, 'grid_map.png')
        shutil.copy(self.object_buffer_fname, self.object_fname)
        
        # load object info
        # with open(self.object_fname, 'r') as f:
        #     object_info = json.load(f)
        
        # load basic info
        with open(self.basic_info_fname, 'r') as f:
            basic_info = json.load(f)

        # load grid map
        if os.path.exists(self.grid_map_fname):
            grid_map = plt.imread(self.grid_map_fname)

        # env basic info
        self.step_time = basic_info['step_time']
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
        with open(self.object_fname, encoding="utf-8") as f:
            for line in f:
                object_info = json.loads(line)
                counts = object_info['count']
                target_visible = object_info['target_visible']
                robot_info = object_info['robot'][0]
                target_info = object_info['target'][0]
                obstacle_info = object_info['obstacles']
                human_info = object_info['humans']

                if target_visible == 0:
                    time_in_search += self.step_time
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
                    time_in_personal_zone += self.step_time

                closest_human_dist = min([np.linalg.norm(np.array(robot_position) - np.array(human[1:3])) for human in human_info]) if len(human_info) > 0 else 100

                robot_human_dist_with_radius = closest_human_dist - (robot_radius + target_radius)


                if robot_human_dist_with_radius < zone_thresholds[0]:
                    time_in_private_zone += self.step_time

        robot_positions = np.array(robot_positions)
        robot_velocities = np.array(robot_velocities)
        robot_target_dists = np.array(robot_target_dists)
        target_positions = np.array(target_positions)
        target_missed = np.array(target_missed)

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

        robot_accelerations = np.diff(robot_velocities, axis=0, prepend=robot_velocities[:1]) / self.step_time
        robot_accelerations = np.linalg.norm(robot_accelerations, axis=1)


        robot_jerk = np.diff(robot_velocities, n=2, axis=0, prepend=robot_velocities[:2]) / self.step_time**2
        robot_jerk = np.linalg.norm(robot_jerk, axis=1)

        # save eval result
        position_fig = os.path.join(self.eval_dir, 'position.png')
        # self.draw_traj(robot_positions, target_positions, position_fig)

        robot_target_dist_fig = os.path.join(self.eval_dir, 'robot_target_dist.png')
        # self.draw_motion_info(robot_target_dists, "Robot Target Distance", "Distance [m]", robot_target_dist_fig)

        robot_velocities = np.linalg.norm(robot_velocities, axis=1)
        velocity_fig = os.path.join(self.eval_dir, 'velocity.png')
        # self.draw_motion_info(robot_velocities, "Robot Velocity", "Velocity [m/s]", velocity_fig)

        acceleration_fig = os.path.join(self.eval_dir, 'acceleration.png')
        # self.draw_motion_info(robot_accelerations, "Robot Acceleration", "Acceleration [m/s^2]", acceleration_fig)

        jerk_fig = os.path.join(self.eval_dir, 'jerk.png')
        # self.draw_motion_info(robot_jerk, "Robot Jerk", "Jerk [m/s^3]", jerk_fig)

        self.eval_result_fname = os.path.join(self.eval_dir, 'eval_result.json')
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
            "total_time": counts*self.step_time,
        }
        with open(self.eval_result_fname, "w") as f:
            json.dump(eval_result, f, indent=4)
    
    def draw_traj(self, robot_positions, target_positions, fname, figsize=(6, 6)):
        """draw robot target trajectory
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(robot_positions[:, 0],  robot_positions[:, 1],
            label="Robot", linewidth=2)
        ax.plot(target_positions[:, 0], target_positions[:, 1],
                label="Target", linewidth=2)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        ax.set_title("Robot & Target Trajectories")
        fig.tight_layout()
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


    def draw_motion_info(self, motion_info, title, y_label, fname, figsize=(6, 6)):
        """draw robot target trajectory
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([i*self.step_time for i in range(len(motion_info))],  motion_info[:],
            label="Robot", linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)



