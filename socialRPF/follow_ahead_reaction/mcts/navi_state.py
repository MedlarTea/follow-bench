import os

import numpy as np

from .follow_task_utils import follow_reward


class navState(object):

    def __init__(self, params, state, next_to_move):
        self.params = params
        self.state = state
        self.next_to_move = next_to_move  # 0: robot's turn,   1: human's turn

    def nextToMove(self):
        return self.next_to_move

    def move(self, action):
        new_state = self.calculate_new_state(action)
        next_to_move = 0 if self.next_to_move == 1 else 1
        return navState(self.params, new_state, next_to_move)

    def calculate_new_state(self, action):
        if self.next_to_move == 0:
            ind = 0
            dist = self.params['robot_vel'] * self.params['dt']
            angle = self.params['robot_angle'] * np.pi / 180
        else:
            ind = 1
            angle = self.params['human_angle'] * np.pi / 180
            dist = self.params['human_vel'] * self.params['dt']

        if action == 'right' or action == 'fast_right':
            angle *= -1.0

        if action == 'straight' or action == 'fast_straight':
            angle *= 0

        if action in {'fast_straight', 'fast_left', 'fast_right'}:
            dist *= self.params['robot_vel_fast_lamda']

        new_s = np.copy(self.state)
        new_s[ind, 0] = self.state[ind, 0] + dist * np.cos(angle + self.state[ind, 2])
        new_s[ind, 1] = self.state[ind, 1] + dist * np.sin(angle + self.state[ind, 2])
        new_s[ind, 2] = angle + self.state[ind, 2]
        return new_s

    def calculate_reward(self, state):
        follow_mode = self.params.get('follow_mode', 'front')
        desired_distance = self.params.get('desired_distance', 1.5)
        rel_vec = state[0, :2] - state[1, :2]
        reward, parts = follow_reward(
            follow_mode,
            desired_distance,
            rel_vec,
            state[1, 2],
            state[0, 2],
        )

        if os.getenv('FOLLOW_AHEAD_DEBUG_REWARD', '0') == '1':
            print(
                '[REWARD] '
                f"mode={follow_mode} "
                f"desired_distance={desired_distance:.3f} "
                f"distance={parts['distance']:.3f} "
                f"diff={parts['diff']:.3f} "
                f"lon={parts['lon']:.3f} "
                f"lat={parts['lat']:.3f} "
                f"lon_err={parts['lon_err']:.3f} "
                f"lat_err={parts['lat_err']:.3f} "
                f"yaw_err={parts['yaw_err'] * 180 / np.pi:.3f} "
                f"r_d={parts['r_d']:.3f} "
                f"r_o={parts['r_o']:.3f} "
                f"shaping={parts['shaping']:.3f} "
                f"total={reward:.3f} "
                f"robot={np.round(state[0], 3).tolist()} "
                f"human={np.round(state[1], 3).tolist()}"
            )

        return reward
