import numpy as np
import random


class DWA():
    def __init__(
        self, 
        robot_tuple: tuple,
        obstacle_radius: float,
        sample_time: float = 0.1, 
        predict_horizon: float = 0.1,
        resolution_scale: tuple[float, float] = (0.01, 0.01),
        weight_heading: float = 2.0,
        weight_obstacle: float = 1.0,
        weight_velocity: float = 1.0,

    ) -> None:

        self.robot_tuple = robot_tuple

        # sample time
        self.dt = sample_time
        # only predict the robot state at the next time step
        self.predict_horizon = predict_horizon

        # robot parameters
        self.robot_width  = robot_tuple.width
        self.robot_length = robot_tuple.length
        self.robot_radius = robot_tuple.radius

        # obstacle radius
        self.obstacle_radius = obstacle_radius

        # diff: [v_min, v_max, w_min, w_max]; omni: [vx_min, vx_max, vy_min, vy_max]
        self.vel_cons = [robot_tuple.min_speed[0], robot_tuple.max_speed[0],
                         robot_tuple.min_speed[1], robot_tuple.max_speed[1]]

        # diff: [v_acc_max, w_acc_max]; omni: [vx_acc_max, vy_acc_max]
        self.acc_cons = robot_tuple.max_acce

        # velocity sampling resolution
        self.vel_resolution = [resolution_scale[0] * robot_tuple.max_speed[0],
                               resolution_scale[1] * robot_tuple.max_speed[1]]

        # cost function weight
        self.weight_heading  = weight_heading
        self.weight_obstacle = weight_obstacle
        self.weight_velocity = weight_velocity

        # deadlock flag
        self.is_deadlocked = False
        self.deadlock_speed_threshold = 0.01

    
    def control(self, robot_pose, robot_vel, goal, obstacles):

        robot_pose = np.array(robot_pose).squeeze()
        robot_vel  = np.array(robot_vel).squeeze()

        # calculate velocity ​​search space
        allowable_vel = self._calc_dynamic_window(robot_vel)

        # generate all candidate trajectories
        candidate_trajectories = self._generate_trajectory(robot_pose, allowable_vel)

        # evaluate all candidate trajectories and select the optimal control command and optimal trajectory
        opt_vel, opt_trajectory = self._evaluate_trajectory(candidate_trajectories, robot_pose, robot_vel, goal, obstacles)

        opt_vel = np.array(opt_vel).reshape(2, 1)
        info = {"arrive": False, "opt_state_list": [np.array([[opt_traj[0]], [opt_traj[1]]]) for opt_traj in opt_trajectory]}

        return opt_vel, info


    def _calc_dynamic_window(self, robot_vel):

        allowable_vel = [min( max( self.vel_cons[0], robot_vel[0] - self.acc_cons[0] * self.dt ), 0 ),
                         max( min( self.vel_cons[1], robot_vel[0] + self.acc_cons[0] * self.dt ), 0 ),
                         min( max( self.vel_cons[2], robot_vel[1] - self.acc_cons[1] * self.dt ), 0 ),
                         max( min( self.vel_cons[3], robot_vel[1] + self.acc_cons[1] * self.dt ), 0 )]

        # handling no-solution situations (such as constraint conflicts)
        if allowable_vel[0] > allowable_vel[1] or allowable_vel[2] > allowable_vel[3]:
            print("[WARN] Dynamic window conflict: forcing stop.")
            return [0, 0, 0, 0]
        
        return allowable_vel
    

    def _generate_trajectory(self, robot_pose, allowable_vel):

        candidate_trajectories = []

        # Compute number of velocity samples along each dimension
        vel_num = [
            int((allowable_vel[1] - allowable_vel[0]) / self.vel_resolution[0]) + 1,
            int((allowable_vel[3] - allowable_vel[2]) / self.vel_resolution[1]) + 1
        ]

        # Sample velocities within the dynamic window and predict trajectories
        for vel_forward in np.linspace(allowable_vel[0], allowable_vel[1], vel_num[0]):
            for vel_lateral in np.linspace(allowable_vel[2], allowable_vel[3], vel_num[1]):
                sample_trajectory = self._trajectory_predict(robot_pose, vel_forward, vel_lateral)
                candidate_trajectories.append(sample_trajectory)

        return candidate_trajectories


    def _trajectory_predict(self, robot_pose, vel_forward, vel_lateral):

        # Only predict the robot's position at the next moment
        current_state = np.hstack([robot_pose, vel_forward, vel_lateral])
        next_state    = self._motion_predict_model(current_state)
        predict_traj  = np.vstack([current_state, next_state])
        
        return predict_traj


    def _motion_predict_model(self, current_state):
        # diff: current_state=[x, y, theta, v, w]
        # omni: current_state=[x, y, theta=0, vx, vy]

        next_state = current_state.copy()

        if self.robot_tuple.kinematics == "diff":
            next_state[0] = current_state[0] + current_state[3] * np.cos(current_state[2]) * self.dt
            next_state[1] = current_state[1] + current_state[3] * np.sin(current_state[2]) * self.dt
            next_state[2] = current_state[2] + current_state[4] * self.dt
            next_state[3] = current_state[3]
            next_state[4] = current_state[4]

        elif self.robot_tuple.kinematics == "omni":
            next_state[0] = current_state[0] + current_state[3] * self.dt
            next_state[1] = current_state[1] + current_state[4] * self.dt
            next_state[3] = current_state[3]
            next_state[4] = current_state[4]

        return next_state


    def _evaluate_trajectory(self, candidate_trajectories, robot_pose, robot_vel, goal, obstacles):

        heading_cost, obstacle_cost, velocity_cost = [], [], []
        valid_trajectories = []

        # remove candidate trajectories that collide with obstacles, then calculate the cost of valid trajectories
        for trajectory in candidate_trajectories:
            
            obs_cost = self._obstacle_cost(trajectory, obstacles)
            if obs_cost == float('inf'):
                continue

            obstacle_cost.append(obs_cost)
            heading_cost.append(self._heading_cost(trajectory, goal))
            velocity_cost.append(self._velocity_cost(trajectory))

            valid_trajectories.append(trajectory)

        # return current pose for in-place rotation when no collision-free trajectory exists
        if not valid_trajectories:

            # Determine stationary velocity based on robot type
            if self.robot_tuple.kinematics == "diff":
                rotation_direction = 1 if random.random() > 0.5 else -1
                stationary_velocity = [0.1, self.vel_cons[3] * rotation_direction]
            elif self.robot_tuple.kinematics == "omni":
                lateral_direction = 1 if random.random() > 0.5 else -1
                stationary_velocity = [0.1, self.vel_cons[3] * lateral_direction]

            # create stationary trajectory by repeating the current pose with velocity
            stationary_trajectory = np.repeat(
                [np.append(robot_pose, stationary_velocity)],
                int(self.predict_horizon / self.dt) + 1,
                axis=0
            )
            
            return stationary_velocity, stationary_trajectory

        # normalize costs
        obs_cost_norm      = self._normalize_cost(np.array(obstacle_cost))
        heading_cost_norm  = self._normalize_cost(np.array(heading_cost))
        velocity_cost_norm = self._normalize_cost(np.array(velocity_cost))

        # evaluate weighted total cost
        min_total_cost = float('inf')
        opt_vel = [0.0, 0.0]
        opt_trajectory = np.array(robot_pose)

        for i in range(len(valid_trajectories)):
            total_cost = (
                self.weight_heading  * heading_cost_norm[i] +
                self.weight_obstacle * obs_cost_norm[i] +
                self.weight_velocity * velocity_cost_norm[i]
            )

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                opt_trajectory = valid_trajectories[i]
                opt_vel = [opt_trajectory[-1, 3], opt_trajectory[-1, 4]]

                # deadlock detection and recovery           
                if self.robot_tuple.kinematics == "diff":
                    # diff: check if forward speed is below threshold
                    self.is_deadlocked = (abs(opt_vel[0]) < self.deadlock_speed_threshold and 
                                          abs(robot_vel[0]) < self.deadlock_speed_threshold)
                elif self.robot_tuple.kinematics == "omni":
                    # omni: check speed magnitude
                    self.is_deadlocked = (np.hypot(opt_vel[0], opt_vel[1]) < self.deadlock_speed_threshold and
                                          np.hypot(robot_vel[0], robot_vel[1]) < self.deadlock_speed_threshold)
                if self.is_deadlocked:
                    opt_vel[1] = -self.vel_cons[3]

        return opt_vel, opt_trajectory


    def _obstacle_cost(self, trajectory, obstacles):

        if len(obstacles) == 0:
            return 0.0

        # Distances to all obstacles
        diff  = obstacles[:, :2] - trajectory[-1, :2]
        dists = np.hypot(diff[:, 0], diff[:, 1])

        # Check for collision
        if np.any(dists <= (self.robot_radius + self.obstacle_radius)):
            return float("Inf")

        min_dist = max(np.min(dists - self.robot_radius - self.obstacle_radius), 1e-6)
        return 1.0 / min_dist
    

    def _heading_cost(self, trajectory, goal):

        if self.robot_tuple.kinematics == "diff":
            # trajectory[-1, 0] and trajectory[-1, 1] are the last position of the trajectory
            dx = goal[0] - trajectory[-1, 0]
            dy = goal[1] - trajectory[-1, 1]

            angle_h2r = np.atan2(dy, dx)
            angle_diff = angle_h2r - trajectory[-1][2]

            heading = abs(np.atan2(np.sin(angle_diff), np.cos(angle_diff)))

        elif self.robot_tuple.kinematics == "omni":

            # trajectory[-1][3] and trajectory[-1][4] are the last velocity components of the trajectory
            vx, vy = trajectory[-1][3], trajectory[-1][4]
            direction = np.array([vx, vy])

            # the direction vector of the target relative to the last position of the trajectory
            x, y = trajectory[-1][0], trajectory[-1][1]
            to_goal = goal[:2].flatten() - np.array([x, y])

            # normalized direction vector
            norm_dir = np.linalg.norm(direction)
            norm_goal = np.linalg.norm(to_goal)

            if norm_dir < 1e-3 or norm_goal < 1e-3:
                # maximum deviation penalty
                return np.pi 
            
            direction /= norm_dir
            to_goal /= norm_goal

            # calculate the dot product of directions and to_goal, and limit the result to the range [-1.0, 1.0]
            cos_phi = np.clip(np.dot(direction, to_goal), -1.0, 1.0)
            # ∈ [0, π]
            heading = np.arccos(cos_phi) 

        return heading
    

    def _velocity_cost(self, trajectory):

        if self.robot_tuple.kinematics == "diff":
            velocity = self.vel_cons[1] - trajectory[-1, 3]

        elif self.robot_tuple.kinematics == "omni":
            vx, vy = trajectory[-1, 3], trajectory[-1, 4]
            V = np.hypot(vx, vy)
            V_max = np.hypot(self.vel_cons[1] , self.vel_cons[3])
            velocity = V_max - abs(V)

        return velocity
    

    def _normalize_cost(self, cost):
        # Min-Max Normalization
        if len(cost) == 0:
            return cost
        max_val = np.max(cost)
        min_val = np.min(cost)
        if max_val == min_val:
            return np.zeros_like(cost)
        return (cost - min_val) / (max_val - min_val)