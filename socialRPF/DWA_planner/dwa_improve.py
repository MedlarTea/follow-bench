import numpy as np

class DWA():
    def __init__(
        self, 
        robot_tuple: tuple,
        obstacle_radius: float,
        sample_time: float = 0.1, 
        predict_horizon: float = 2.0,
        resolution_scale: tuple[float, float] = (0.02, 0.02),
        weight_heading: float = 3.0,
        weight_distance: float = 2.0,
        weight_obstacle: float = 1.0,
        weight_velocity: float = 0.5
    ) -> None:  
        
        # Simulation parameters
        self.dt              = sample_time
        self.predict_horizon = predict_horizon
        self.robot_tuple     = robot_tuple
        self.robot_width     = robot_tuple.width
        self.robot_length    = robot_tuple.length
        self.robot_radius    = robot_tuple.radius
        self.obstacle_radius = obstacle_radius

        # Velocity constraints
        # diff-drive: [v_min, v_max, w_min, w_max]; omni-drive: [vx_min, vx_max, vy_min, vy_max]
        self.vel_cons = [
            robot_tuple.min_speed[0], robot_tuple.max_speed[0],
            robot_tuple.min_speed[1], robot_tuple.max_speed[1]
        ]
        
        # Acceleration constraints
        # diff-drive: [v_acc_max, w_acc_max]; omni-drive: [vx_acc_max, vy_acc_max]
        self.acc_cons = robot_tuple.max_acce

        # Velocity sampling resolution
        self.vel_resolution = [
            resolution_scale[0] * robot_tuple.max_speed[0],
            resolution_scale[1] * robot_tuple.max_speed[1]
        ]

        # Weights for cost functions
        self.weight_heading  = weight_heading
        self.weight_distance = weight_distance
        self.weight_obstacle = weight_obstacle
        self.weight_velocity = weight_velocity

        # Deadlock detection parameters
        self.is_deadlocked = False
        self.deadlock_speed_threshold = 0.01


    def control(self, robot_pose, robot_vel, goal_traj, obstacles):
        
        robot_pose = np.array(robot_pose).squeeze()
        robot_vel  = np.array(robot_vel).squeeze()
        goal_traj  = np.array(goal_traj)
        obstacles  = np.array(obstacles)

        # Step 1: Compute feasible velocity search space
        allowable_vel = self._calc_dynamic_window(robot_vel)

        # Step 2: Generate all candidate trajectories
        candidate_trajectory = self._generate_trajectory(robot_pose, allowable_vel)

        # Step 3: Evaluate trajectories and pick the best
        opt_vel, opt_trajectory, all_trajectories = self._evaluate_trajectory(candidate_trajectory, robot_vel, goal_traj, obstacles)
        
        # Format results
        opt_vel = np.array(opt_vel).reshape(2, 1)
        opt_state_list = opt_trajectory[:, :2].T
        all_trajectories_list = [traj[:, :2].T for traj in all_trajectories
                                 if not np.array_equal(traj, opt_trajectory)]
        info = {
            "arrive": False, 
            "opt_state_list": opt_state_list, 
            "all_trajectories_list": all_trajectories_list
        }

        return opt_vel, info
    

    def _calc_dynamic_window(self, robot_vel):
        """
        Compute the dynamic window for the robot's velocities.

        Parameters:
            robot_vel (array-like): Current robot velocity.
                - Differential drive: [v, w] (linear and angular velocities)
                - Omnidirectional: [vx, vy] (x and y velocities)

        Returns:
            allowable_vel (np.ndarray): Feasible velocity range.
                - Differential drive: [v_min, v_max, w_min, w_max]
                - Omnidirectional: [vx_min, vx_max, vy_min, vy_max]
                If there is a conflict (lower bound > upper bound), returns an array of zeros.
        """
        allowable_vel = np.array([
            max(self.vel_cons[0], robot_vel[0] - self.acc_cons[0] * self.dt),
            min(self.vel_cons[1], robot_vel[0] + self.acc_cons[0] * self.dt),
            max(self.vel_cons[2], robot_vel[1] - self.acc_cons[1] * self.dt),
            min(self.vel_cons[3], robot_vel[1] + self.acc_cons[1] * self.dt)
        ], dtype=float)
        
        if np.any(allowable_vel[::2] > allowable_vel[1::2]):
            # print("[WARN] Dynamic window conflict: forcing stop.")
            return np.zeros(4, dtype=float)
        
        return allowable_vel
    

    def _generate_trajectory(self, robot_pose, allowable_vel):
        """
        Generate candidate trajectories for all velocity combinations within the dynamic window.

        Parameters:
            robot_pose (array-like): Current robot pose [x, y, yaw].
            allowable_vel (array-like): Dynamic window of velocities 

        Returns:
            candidate_trajectory (np.ndarray): Array of shape (N_traj, steps, 5) containing
                candidate trajectories. Each trajectory has [x, y, yaw, v_forward/vx, v_lateral/w] at each timestep.
        """

        # Number of velocity samples
        v_num = [
            int((allowable_vel[1] - allowable_vel[0]) / self.vel_resolution[0]) + 1,
            int((allowable_vel[3] - allowable_vel[2]) / self.vel_resolution[1]) + 1
        ]

        # Sample velocities
        v_forwards = np.linspace(allowable_vel[0], allowable_vel[1], v_num[0])
        v_laterals = np.linspace(allowable_vel[2], allowable_vel[3], v_num[1])

        # Create a meshgrid and flatten to get all combinations of forward and lateral velocities
        v_forward, v_lateral = np.meshgrid(v_forwards, v_laterals, indexing='ij')
        v_forward = v_forward.flatten()  # (N_traj,)
        v_lateral = v_lateral.flatten()  # (N_traj,)
        N_traj = len(v_forward)

        # Initialize trajectory array
        steps = int(self.predict_horizon / self.dt) + 1
        candidate_trajectory = np.zeros((N_traj, steps, 5))
        candidate_trajectory[:, 0, :3] = robot_pose[:3]
        candidate_trajectory[:, 0, 3]  = v_forward
        candidate_trajectory[:, 0, 4]  = v_lateral

        # Forward simulate each trajectory
        if self.robot_tuple.kinematics == "diff":
            for t in range(1, steps):
                prev  = candidate_trajectory[:, t-1, :]
                x_new = prev[:,0] + prev[:,3] * np.cos(prev[:,2]) * self.dt
                y_new = prev[:,1] + prev[:,3] * np.sin(prev[:,2]) * self.dt
                yaw_new = prev[:,2] + prev[:,4] * self.dt
                candidate_trajectory[:, t, 0] = x_new
                candidate_trajectory[:, t, 1] = y_new
                candidate_trajectory[:, t, 2] = yaw_new
                candidate_trajectory[:, t, 3] = prev[:,3] 
                candidate_trajectory[:, t, 4] = prev[:,4]
        elif self.robot_tuple.kinematics == "omni":
            for t in range(1, steps):
                prev = candidate_trajectory[:, t-1, :]
                candidate_trajectory[:, t, 0] = prev[:,0] + prev[:,3] * self.dt
                candidate_trajectory[:, t, 1] = prev[:,1] + prev[:,4] * self.dt
                candidate_trajectory[:, t, 2] = prev[:,2]
                candidate_trajectory[:, t, 3] = prev[:,3]
                candidate_trajectory[:, t, 4] = prev[:,4]

        return candidate_trajectory

        
    def _evaluate_trajectory(self, candidate_trajectory, robot_vel, goal_traj, obstacles):
        """
        Evaluate all candidate trajectories and select the optimal one based on cost functions.

        Parameters:
            candidate_trajectory (np.ndarray): Array of shape (N_traj, T, 5), each trajectory has
                                            [x, y, yaw, v_forward, v_lateral] per timestep.
            robot_vel (array-like): Current robot velocity [v, w] or [vx, vy].
            goal_traj (array-like): Goal trajectory for heading evaluation.
            obstacles (array-like): Positions of obstacles.

        Returns:
            opt_vel (list): Optimal velocity [v, w] or [vx, vy] at the last timestep of the optimal trajectory.
            opt_traj (np.ndarray): Optimal trajectory of shape (T, 5).
            candidate_trajectory (np.ndarray): All candidate trajectories (unchanged input, for reference).
        """

        # Compute costs for each trajectory
        obstacle_cost = self._obstacle_cost(candidate_trajectory, obstacles)
        heading_cost  = self._heading_cost(candidate_trajectory, goal_traj)
        distance_cost = self._distance_cost(candidate_trajectory, goal_traj)
        velocity_cost = self._velocity_cost(candidate_trajectory)

        # Normalize costs to [0,1] for fair weighting
        heading_cost  = self._normalize_cost(heading_cost)
        distance_cost = self._normalize_cost(distance_cost)
        velocity_cost = self._normalize_cost(velocity_cost)
        obstacle_cost = self._normalize_cost(obstacle_cost)

        # Weighted sum of costs
        total_costs = (self.weight_heading  * heading_cost  +
                       self.weight_distance * distance_cost +
                       self.weight_obstacle * obstacle_cost +
                       self.weight_velocity * velocity_cost)
        
        # Pick the trajectory with minimum total cost
        opt_idx  = np.argmin(total_costs)
        opt_traj = candidate_trajectory[opt_idx]
        opt_vel  = [opt_traj[-1, 3], opt_traj[-1, 4]]

        # Deadlock detection
        if self.robot_tuple.kinematics == "diff":
            self.is_deadlocked = (abs(opt_vel[0]) < self.deadlock_speed_threshold and
                                  abs(robot_vel[0]) < self.deadlock_speed_threshold)
        elif self.robot_tuple.kinematics == "omni":
            self.is_deadlocked = (np.hypot(opt_vel[0], opt_vel[1]) < self.deadlock_speed_threshold and
                                  np.hypot(robot_vel[0], robot_vel[1]) < self.deadlock_speed_threshold)
            
        # Recovery strategy: if deadlocked, reverse lateral/angular velocity
        if self.is_deadlocked:
            opt_vel[1] = -self.vel_cons[3]

        return opt_vel, opt_traj, candidate_trajectory


    def _heading_cost(self, candidate_trajectory, goal_traj, alpha=0.4):
        """
        Compute heading cost for candidate trajectories.

        Heading cost consists of short-term and long-term components:
        - Short-term: difference between first predicted pose and start of goal trajectory.
        - Long-term: difference between final predicted pose and end of goal trajectory.

        Parameters:
            candidate_trajectory (np.ndarray): shape (N_traj, T, 5)
                Candidate trajectories with [x, y, yaw, vx, vy] for omni or [x, y, yaw, v, w] for diff.
            goal_traj (array-like): shape (T_goal, >=2)
                Reference goal trajectory.
            alpha (float): weight for short-term vs long-term heading.

        Returns:
            heading_cost (np.ndarray): shape (N_traj,)
                Cost for each trajectory, smaller is better.
        """

        # Short-term heading cost
        goal_start = goal_traj[0,:2]
        first_pose = candidate_trajectory[:,1,:2]
        dx_s = goal_start[0] - first_pose[:,0]
        dy_s = goal_start[1] - first_pose[:,1]

        if self.robot_tuple.kinematics == "diff":
            angle_h2r_s  = np.arctan2(dy_s, dx_s)
            angle_diff_s = angle_h2r_s - candidate_trajectory[:,1,2]
            short_term   = np.abs(np.arctan2(np.sin(angle_diff_s), np.cos(angle_diff_s)))
        elif self.robot_tuple.kinematics == "omni":
            vx_s = candidate_trajectory[:,1,3]
            vy_s = candidate_trajectory[:,1,4]
            norm_dir_s  = np.hypot(vx_s, vy_s)
            to_goal_s   = np.vstack([dx_s, dy_s]).T
            norm_goal_s = np.linalg.norm(to_goal_s, axis=1)
            direction_s = np.vstack([vx_s, vy_s]).T / np.maximum(norm_dir_s[:,None],1e-6)
            to_goal_normed_s = to_goal_s / np.maximum(norm_goal_s[:,None],1e-6)
            cos_angle_s = np.clip(np.sum(direction_s * to_goal_normed_s, axis=1), -1.0, 1.0)
            short_term  = np.arccos(cos_angle_s)
            short_term[(norm_dir_s<1e-3)|(norm_goal_s<1e-6)] = np.pi/2

        # Long-term heading cost
        goal_end  = goal_traj[-1,:2]
        last_pose = candidate_trajectory[:,-1,:2]
        dx_l = goal_end[0] - last_pose[:,0]
        dy_l = goal_end[1] - last_pose[:,1]

        if self.robot_tuple.kinematics == "diff":
            angle_h2r_l  = np.arctan2(dy_l, dx_l)
            angle_diff_l = angle_h2r_l - candidate_trajectory[:,-1,2]
            long_term    = np.abs(np.arctan2(np.sin(angle_diff_l), np.cos(angle_diff_l)))
        elif self.robot_tuple.kinematics == "omni":
            vx_l = candidate_trajectory[:,-1,3]
            vy_l = candidate_trajectory[:,-1,4]
            norm_dir_l  = np.hypot(vx_l, vy_l)
            to_goal_l   = np.vstack([dx_l, dy_l]).T
            norm_goal_l = np.linalg.norm(to_goal_l, axis=1)
            direction_l = np.vstack([vx_l, vy_l]).T / np.maximum(norm_dir_l[:,None],1e-6)
            to_goal_normed_l = to_goal_l / np.maximum(norm_goal_l[:,None],1e-6)
            cos_angle_l = np.clip(np.sum(direction_l * to_goal_normed_l, axis=1), -1.0, 1.0)
            long_term   = np.arccos(cos_angle_l)
            long_term[(norm_dir_l<1e-3)|(norm_goal_l<1e-6)] = np.pi/2

        # Weighted sum of short-term and long-term
        heading_cost = alpha*short_term + (1-alpha)*long_term

        return heading_cost


    def _obstacle_cost(self, candidate_trajectory, obstacles):
        """
        Compute obstacle cost for candidate trajectories.

        Cost is based on minimum distance to obstacles along trajectory.
        Trajectories starting in collision are assigned a very large cost.

        Parameters:
            candidate_trajectory (np.ndarray): shape (N_traj, T, 5)
            obstacles (np.ndarray): shape (N_obs, 2) or (N_obs, 3) if including radius

        Returns:
            weighted_cost (np.ndarray): shape (N_traj,)
                Higher cost for closer proximity to obstacles.
        """
        N_traj, T, _ = candidate_trajectory.shape

        if len(obstacles) == 0:
            return np.zeros(N_traj)

        collision_dist = self.robot_radius + self.obstacle_radius

        # Distance to all obstacles at each timestep
        diffs = candidate_trajectory[:, :, None, :2] - obstacles[None, None, :, :2]  # (N_traj, T, N_obs, 2)
        dists = np.hypot(diffs[..., 0], diffs[..., 1])  # (N_traj, T, N_obs)

        # Mark trajectories that start in collision
        initial_collision = np.any(dists[:, 0, :] <= collision_dist, axis=1)
        
        # Minimum distance along trajectory (excluding t=0)
        min_dists_per_point = np.min(dists[:, 1:, :], axis=2) - collision_dist  # (N_traj, T-1)
        min_dists_per_point = np.clip(min_dists_per_point, 1e-6, None)

        # Weighted inverse distance cost
        weights = np.linspace(0.5, 1.0, T-1)
        weighted_cost = np.sum(weights * (1.0 / min_dists_per_point), axis=1) / np.sum(weights)

        # Large penalty for initial collision
        weighted_cost[initial_collision] = 1e6

        return weighted_cost

    
    def _velocity_cost(self, candidate_trajectory):
        """
        Compute velocity cost to encourage faster trajectories.

        Parameters:
            candidate_trajectory (np.ndarray): shape (N_traj, T, 5)

        Returns:
            velocity_cost (np.ndarray): shape (N_traj,)
                Smaller value corresponds to higher velocity (preferred).
        """
            
        if self.robot_tuple.kinematics == "diff":
            return self.vel_cons[1] - candidate_trajectory[:, -1, 3]
        
        elif self.robot_tuple.kinematics == "omni":
            vx = candidate_trajectory[:, -1, 3]
            vy = candidate_trajectory[:, -1, 4]
            V  = np.hypot(vx, vy)
            V_max = np.hypot(self.vel_cons[1], self.vel_cons[3])
            return V_max - V


    def _distance_cost(self, candidate_trajectory, goal_traj):
        """
        Compute distance cost for candidate trajectories based on first predicted point.

        Parameters:
            candidate_trajectory (np.ndarray): shape (N_traj, T, 5)
                Candidate trajectories.
            goal_traj (np.ndarray): shape (T_goal, 2)
                Goal trajectory (x, y). Only first point is used.

        Returns:
            dist_cost (np.ndarray): shape (N_traj,)
                Smaller cost if first predicted point is closer to goal.
        """
        
        # Take first predicted point of each trajectory
        first_pred = candidate_trajectory[:, 1, :2]  # shape (N_traj, 2)
        goal_point = goal_traj[0, :2]                # shape (2,)

        # Euclidean distance
        dx = first_pred[:, 0] - goal_point[0]
        dy = first_pred[:, 1] - goal_point[1]
        dist_cost = np.hypot(dx, dy)

        return dist_cost


    def _normalize_cost(self, cost):
        """
        Normalize cost to [0,1] range for fair weighting between different cost components.

        Parameters:
            cost (array-like): 1D array of costs.

        Returns:
            normalized_cost (np.ndarray): 1D array with values in [0,1]
        """
        if len(cost) == 0:
            return np.array([])
        cost = np.asarray(cost)
        min_val, max_val = cost.min(), cost.max()
        if max_val == min_val:
            return np.zeros_like(cost)
        return (cost - min_val) / (max_val - min_val)