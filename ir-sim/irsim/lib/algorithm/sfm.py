import numpy as np
from numpy.linalg import norm
from math import sin, cos, atan2, asin, pi
from irsim.global_param.world_param import step_time
from irsim.util.util import xy_to_coord, coord_to_xy
from irsim.global_param import env_param

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
        state: list,
        obs_state_list=[],
        goal=None,
        **kwargs
    ):

        self.state = state
        self.obs_state_list = obs_state_list
        self.vxmax = kwargs.get("vxmax", 1.5)
        self.vymax = kwargs.get("vymax", 1.5)
        self.v_max = max(self.vxmax, self.vymax)
        self.acceler = kwargs.get("acceler", 1.0)
        self.goal = np.array(goal).squeeze()[:2]

        # print("[SMF] vxmax", vxmax)
        # print("[SMF] vymax", vymax)

        self.radius = self.state[4]
        self.dt = step_time

        # hyperparameters
        self.gain_k = kwargs.get("gain_k", 1)
        self.gain_a_static = kwargs.get("gain_a_static", 4)
        self.gain_b_static = kwargs.get("gain_b_static", 0.25)

        self.gain_a_agent = kwargs.get("gain_a_agent", 7)
        self.gain_b_agent = kwargs.get("gain_b_agent", 0.5)

        self.avoid_dis = kwargs.get("avoid_dis", 5)  # 安全作用半径
        self.reach_dis = 2*self.radius  # 目标点作用半径
        self.safe_dist = kwargs.get("safe_dist", 0.3)

        # self.goal_threshold = self.radius

        self.fov = False

    def update(self, state, obs_state_list):

        self.state = state
        self.obs_state_list = obs_state_list
    
    def cal_vel(self):
        ego_pos = np.array(self.state[:2])
        ego_vel = np.array(self.state[2:4])

        # print("[SFM] ego_pos:", ego_pos)
        # print("[SFM] ego_vel:", ego_vel)
        
        obs_rel_pos = np.zeros((len(self.obs_state_list), 2))
        obs_rel_dist = np.zeros(len(self.obs_state_list))
        # obs_rel_vel = np.zeros((len(self.obs_state_list), 2))
        for i, obs_state in enumerate(self.obs_state_list):
            obs_rel_pos[i, :] = ego_pos - obs_state[:2]
            obs_rel_dist[i] = norm(obs_rel_pos[i])
            # obs_rel_vel[i, :] = ego_vel - obs_state[2:4]

        # pos, cord_int, vel = agent_state
        nbrs_relpos, nbrs_dis = self.fov_filter(ego_vel, obs_rel_pos, obs_rel_dist) if self.fov else obs_rel_pos, obs_rel_dist
        # x, y = cord_int  # for obstacle maps
        
        # ===== 1. 计算目标吸引力 =====
        direction_to_goal = self.goal - ego_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal > self.reach_dis:
            attractive_force = self.v_max * direction_to_goal/distance_to_goal
        else:
            # print('[SFM] !!! Reach')
            attractive_force = np.zeros(2)
        
        
        # ===== 2. 计算障碍排斥力 =====

        ### repulsive force from obstacle maps
        h, w = env_param.gm_size
        coord_x, coord_y = xy_to_coord([ego_pos[0], ego_pos[1]], env_param.gm_res, env_param.gm_size)
        # 计算切片边界（防止越界）
        coord_y = max(0, min(coord_y, h - 1))
        coord_x = max(0, min(coord_x, w - 1))
        dist = env_param.planner.distance_field[coord_x, coord_y]/4

        # print("dist:", dist)
        
        if dist < self.avoid_dis:
            # 获取梯度方向
            # grad = -np.array([env_param.planner.gradient_field[0, coord_x, coord_y], env_param.planner.gradient_field[1, coord_x, coord_y]])
            grad = np.array([env_param.planner.gradient_field[1, coord_x, coord_y], env_param.planner.gradient_field[0, coord_x, coord_y]])
            
            # 排斥力计算（距离越近力越大）
            # repulsive_force = self.repulsive_gain * (1/dist - 1/self.safety_radius) * grad
            repulsive_force = self.gain_a_static * np.exp( (0.5*self.safe_dist - dist)/ self.gain_b_static ) * grad
        else:
            repulsive_force = np.zeros(2)
        
        ### repulsive force from humans
        if len(nbrs_relpos) != 0:
            
            nbrs_dis = nbrs_dis.reshape(-1,1)
            nbrs_relpos = nbrs_relpos/nbrs_dis
            
            interact_force = self.gain_a_agent * np.exp( (self.safe_dist - nbrs_dis)/ self.gain_b_agent ) * nbrs_relpos
            interact_force = np.sum(interact_force, axis=0)
        else:
            interact_force = np.zeros(2)

        # ===== 合力合成 =====
        # Sum of push & pull forces        
        d_vel = self.gain_k * (attractive_force - ego_vel)
        interaction_vel = repulsive_force + interact_force  # with obstacle maps
        # interaction_vel = interact_force  # with dynamic forces only
        total_d_vel = (d_vel + interaction_vel) * self.dt
        new_vel = ego_vel + total_d_vel
        # print('[SFM] *** output', np.linalg.norm(d_vel), repulsive_force, interact_force, np.linalg.norm(interaction_vel))
        # print("[SFM] self.goal:", self.goal)
        # print("[SFM] attractive_force:", attractive_force)
        # print("[SFM] d_vel:", d_vel)
        # print("[SFM] repulsive_force:", repulsive_force)
        # print("[SFM] interact_force:", interact_force)
        # print("[SFM] ego_goal:", self.goal)
        # print("[SFM] ego_vel:", ego_vel)
        # print("[SFM] total_d_vel:", total_d_vel)
        # print("[SFM] new_vel:", new_vel)

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        # if distance_to_goal < self.goal_threshold:
            # return np.zeros(2)

        act_norm = np.linalg.norm(new_vel)

        # if act_norm < 0.1 and np.linalg.norm(attractive_force) > 0.1:
        #     print("[SFM] attractive_force:", attractive_force)
        #     print("[SFM] repulsive_force:", repulsive_force)
        #     print("[SFM] interact_force:", interact_force)
        #     print("[SFM] total_d_vel:", total_d_vel)
        #     print("[SFM] new_vel:", new_vel)

        if act_norm > self.v_max:
            return new_vel*self.v_max / act_norm
        else:
            return new_vel

    @staticmethod
    def compute_gradient(distance_field):
        """计算EDT的梯度场"""
        
        grad = np.gradient(distance_field)
        
        # 计算梯度模长
        magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
        
        # 处理零梯度区域
        magnitude[magnitude == 0] = 1e-6  # 避免除以零
        
        # 归一化梯度
        grad_norm = grad / magnitude
        
        return grad_norm
    
    @staticmethod
    def fov_filter(curr_dir, obs_rel_pos, obs_rel_dist):

        if len(obs_rel_dist) == 0:
            nbrs_pos = []
            nbrs_dist = []

            return nbrs_pos, nbrs_dist

        cos_angles = np.dot(-obs_rel_pos, curr_dir)

        # 筛选在视野范围内的邻居（夹角在±90度内，即cos值>0）
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
    