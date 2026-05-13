import numpy as np
from numpy.linalg import norm
from math import sin, cos, atan2, asin, pi

# state: [x, y, vx, vy, radius, vx_des, vy_des]
# obs_state_list: [[x, y, vx, vy, radius]]
# rvo_vel: [vx, vy]


class social_force_model:
    """
    A class to implement the ORCA (Optimal Reciprocal Collision Avoidance) algorithm.

    Args:

        state (list): The rvo state of the agent [x, y, vx, vy, radius, vx_des, vy_des].
        obs_state_list (list) : List of states of static obstacles [[x, y, vx, vy, radius]].
        vxmax (float): Maximum velocity in the x direction.
        vymax (float): Maximum velocity in the y direction.
        acceler (float): Acceleration limit.
    """

    def __init__(
        self,
        radius,
        sample_time,
        predict_horizon,
        **kwargs,
    ):
        self.vxmax = kwargs.get("vxmax", 1.5)
        self.vymax = kwargs.get("vymax", 1.5)
        self.v_max = max(self.vxmax, self.vymax)
        self.acceler = kwargs.get("acceler", 1.0)
        # self.goal = np.array(goal).squeeze()[:2]

        # print("[SMF] vxmax", vxmax)
        # print("[SMF] vymax", vymax)

        self.radius = radius
        self.dt = sample_time
        self.predict_horizon = predict_horizon

        # hyperparameters
        self.gain_k = kwargs.get("gain_k", 1)
        self.gain_a_static = kwargs.get("gain_a_static", 4)
        self.gain_b_static = kwargs.get("gain_b_static", 0.25)

        self.gain_a_agent = kwargs.get("gain_a_agent", 7)
        self.gain_b_agent = kwargs.get("gain_b_agent", 0.5)

        self.avoid_dis = kwargs.get("avoid_dis", 2.0)  # only obstacles within this range are considered (unit: m)
        self.reach_dis = 2*self.radius  # effective radius around the goal point
        self.safe_dist = kwargs.get("safe_dist", 0.3)

        # self.goal_threshold = self.radius

        self.fov = False

        self.last_goal = None

    def control(self, robot_pose, robot_vel, goal, static_obstacles, humans):
        ego_pos = np.array(robot_pose).squeeze()[:2]  # not consider orientation
        ego_vel = np.array(robot_vel).squeeze()
        goal = np.array(goal).squeeze()[:2]  # not consider orientation

        # print("[SFM] ego_pos:", ego_pos)
        # print("[SFM] ego_vel:", ego_vel)
        # print("[SFM] goal:", goal)


        # ===== 1. Compute the goal-attraction force =====
        direction_to_goal = goal - ego_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)

        # print("[SFM]")
        
        if distance_to_goal > self.reach_dis:
            attractive_force = self.v_max * direction_to_goal/distance_to_goal
        else:
            # print('[SFM] !!! Reach')
            attractive_force = np.zeros(2)
        
        # ===== 2. Compute the pedestrian repulsion force =====
        humans_rel_pos = np.zeros((len(humans), 2))
        humans_rel_dist = np.zeros(len(humans))

        for i, human_state in enumerate(humans):
            humans_rel_pos[i, :] = ego_pos - human_state[:2]
            humans_rel_dist[i] = norm(humans_rel_pos[i])
        
        # pos, cord_int, vel = agent_state
        nbrs_relpos, nbrs_dis = self.fov_filter(ego_vel, humans_rel_pos, humans_rel_dist) if self.fov else humans_rel_pos, humans_rel_dist

        if len(nbrs_relpos) != 0:
            nbrs_dis = nbrs_dis.reshape(-1,1)
            nbrs_relpos = nbrs_relpos/nbrs_dis
            
            interact_force = self.gain_a_agent * np.exp( (self.safe_dist - nbrs_dis)/ self.gain_b_agent ) * nbrs_relpos
            interact_force = np.sum(interact_force, axis=0)
        else:
            interact_force = np.zeros(2)

        # ===== 3. Compute the obstacle repulsion force =====
        obs_rel_pos = np.zeros((len(static_obstacles), 2))
        obs_rel_dist = np.zeros(len(static_obstacles))
        for i, obs_state in enumerate(static_obstacles):
            # print(ego_pos, obs_state)
            obs_rel_pos[i, :] = ego_pos - obs_state[:2]
            obs_rel_dist[i] = norm(obs_rel_pos[i])

        obs_rel_pos = obs_rel_pos[obs_rel_dist<self.avoid_dis]
        obs_rel_dist = obs_rel_dist[obs_rel_dist<self.avoid_dis]

        if len(obs_rel_pos) != 0:
            obs_rel_pos = obs_rel_pos/norm(obs_rel_pos)
            obs_rel_dist = obs_rel_dist.reshape(-1,1)
            
            repulsive_force = self.gain_a_static * np.exp( (self.safe_dist - obs_rel_dist)/ self.gain_b_static ) * obs_rel_pos
            repulsive_force = np.sum(repulsive_force, axis=0)
            # repulsive_force = self.gain_a_static * np.exp( (0.5*self.safe_dist - dist)/ self.gain_b_static ) * grad
        else:
            repulsive_force = np.zeros(2)
        
        # ===== Sum of all forces =====
        # Sum of push & pull forces        
        # print("[SFM] attractive_force:", attractive_force)
        # print("[SFM] ego_vel:", ego_vel)
        d_vel = self.gain_k * (attractive_force - ego_vel)
        interaction_vel = repulsive_force + interact_force  # with obstacle maps
        # interaction_vel = interact_force  # with dynamic forces only
        total_d_vel = (d_vel + interaction_vel) * self.dt
        opt_vel = ego_vel + total_d_vel
        # print('[SFM] *** output', np.linalg.norm(d_vel), repulsive_force, interact_force, np.linalg.norm(interaction_vel))
        # print("[SFM] self.goal:", self.goal)
        # print("[SFM] d_vel:", d_vel)
        # print("[SFM] repulsive_force:", repulsive_force)
        # print("[SFM] interact_force:", interact_force)
        # print("[SFM] ego_goal:", self.goal)
        # print("[SFM] ego_vel:", ego_vel)
        # print("[SFM] total_d_vel:", total_d_vel)
        # print("[SFM] opt_vel:", opt_vel)

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref

        act_norm = np.linalg.norm(opt_vel)
        if act_norm > self.v_max:
            opt_vel = opt_vel*self.v_max / act_norm  # vx, vy
        
        # if distance_to_goal < 0.1:
        #     opt_vel = np.zeros(2)

        future_steps = int(self.predict_horizon/self.dt)
        opt_trajectory = np.zeros((future_steps, 2))
        opt_trajectory[0, 0] = ego_pos[0]
        opt_trajectory[0, 1] = ego_pos[1]
        for i in range(1, future_steps):
            opt_trajectory[i, 0] = opt_trajectory[i-1, 0] + opt_vel[0] * self.dt
            opt_trajectory[i, 1] = opt_trajectory[i-1, 1] + opt_vel[1] * self.dt
        
        if self.last_goal is not None:
            goal_change = np.linalg.norm(goal - self.last_goal)
            if goal_change < 0.05 and distance_to_goal < 0.05:
                opt_vel = np.zeros(2)

        opt_vel = np.array(opt_vel).reshape(2, 1)
        info = {"arrive": False, "opt_state_list": [np.array([[opt_traj[0]], [opt_traj[1]]]) for opt_traj in opt_trajectory]}

        self.last_goal = goal

        return opt_vel, info
    
    @staticmethod
    def compute_gradient(distance_field):
        """Compute the gradient field of an EDT (Euclidean distance transform)."""

        grad = np.gradient(distance_field)

        # Magnitude of the gradient
        magnitude = np.sqrt(grad[0]**2 + grad[1]**2)

        # Handle zero-gradient regions
        magnitude[magnitude == 0] = 1e-6  # avoid division by zero

        # Normalize the gradient
        grad_norm = grad / magnitude
        
        return grad_norm
    
    @staticmethod
    def fov_filter(curr_dir, obs_rel_pos, obs_rel_dist):

        if len(obs_rel_dist) == 0:
            nbrs_pos = []
            nbrs_dist = []

            return nbrs_pos, nbrs_dist

        cos_angles = np.dot(-obs_rel_pos, curr_dir)

        # Keep neighbors within the field of view (angle within +/-90 deg, i.e. cos > 0)
        in_fov = cos_angles >= 0

        fov_nbrs_dist = obs_rel_pos[in_fov]
        fov_nbrs_relpos = obs_rel_dist[in_fov]
        
        if len(fov_nbrs_dist) > 5:
            sorted_indices = np.argsort(fov_nbrs_dist)[:5]          
            nbrs_pos = fov_nbrs_relpos[sorted_indices]
            nbrs_dist = fov_nbrs_dist[sorted_indices]
        else:
            nbrs_pos = fov_nbrs_relpos
            nbrs_dist = fov_nbrs_dist

        return nbrs_pos, obs_rel_pos
    