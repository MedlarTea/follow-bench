import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SOURCE_ROOT = Path(__file__).resolve().parents[2]
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from follow_ahead_reaction.mcts.follow_task_utils import (
    build_rl_observation,
    desired_local_point,
    follow_reward,
    local_diagnostics,
    wrap_to_pi,
)


class FollowTaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        follow_mode="back",
        desired_distance=1.5,
        max_steps=100,
        world_size=10.0,
        distance_threshold=1e-9,
        init_radius_min=None,
        init_radius_max=None,
        seed=None,
        obs_mode="relative_pose",
    ):
        super().__init__()
        self.follow_mode = follow_mode
        self.desired_distance = float(desired_distance)
        self.max_steps = int(max_steps)
        self.world_size = float(world_size)
        self.distance_threshold = float(distance_threshold)
        self.init_radius_min = float(init_radius_min if init_radius_min is not None else max(0.5, self.desired_distance - 0.5))
        self.init_radius_max = float(init_radius_max if init_radius_max is not None else self.desired_distance + 1.0)
        self.obs_mode = obs_mode
        self.human_step_size = 0.5
        self.robot_step_size_slow = 0.5
        self.robot_step_size_fast = 1.0

        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self._heading_table = np.array([0.0, np.pi * 7 / 4, np.pi / 2, np.pi * 3 / 4, np.pi, np.pi * 5 / 4, np.pi * 3 / 2, np.pi / 4], dtype=np.float32)
        self._reset_heading_table = np.array([0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi, np.pi * 5 / 4, np.pi * 3 / 2, np.pi * 7 / 4], dtype=np.float32)
        self._human_turn_table = np.array([0.0, np.pi / 8, -np.pi / 8, np.pi / 4, -np.pi / 4], dtype=np.float32)

        self.seed_value = seed
        self.human_position = None
        self.robot_position = None
        self.human_orientation = 0.0
        self.robot_orientation = 0.0
        self.step_count = 0
        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed

        margin = self.init_radius_max + self.robot_step_size_fast + 0.5
        low = max(margin, 0.0)
        high = max(self.world_size - margin, low + 1e-6)
        self.human_position = self.np_random.uniform(low, high, size=2).astype(np.float32)
        self.human_orientation = 0.0

        desired_lon, desired_lat = desired_local_point(self.follow_mode, self.desired_distance)
        noise_radius = float(self.np_random.uniform(0.0, max(0.6, self.init_radius_max - self.init_radius_min)))
        noise_angle = float(self.np_random.uniform(-np.pi, np.pi))
        local_noise = np.array([noise_radius * np.cos(noise_angle), noise_radius * np.sin(noise_angle)], dtype=np.float32)
        desired_rel = np.array([desired_lon, desired_lat], dtype=np.float32)
        self.robot_position = self._clip_position(self.human_position + desired_rel + local_noise)
        self.robot_orientation = float(self.np_random.choice(self._reset_heading_table))
        self.step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        heading = float(self._heading_table[action % 8])
        step_size = self.robot_step_size_slow if action < 8 else self.robot_step_size_fast

        self.robot_orientation = heading
        self.robot_position = self._clip_position(
            self.robot_position + np.array([step_size * np.cos(self.robot_orientation), step_size * np.sin(self.robot_orientation)], dtype=np.float32)
        )

        reward, _ = follow_reward(
            self.follow_mode,
            self.desired_distance,
            self.robot_position - self.human_position,
            self.human_orientation,
            self.robot_orientation,
        )

        self.human_orientation = float(wrap_to_pi(self.human_orientation + float(self.np_random.choice(self._human_turn_table))))
        self.human_position = self._clip_position(
            self.human_position
            + np.array([
                self.human_step_size * np.cos(self.human_orientation),
                self.human_step_size * np.sin(self.human_orientation),
            ], dtype=np.float32)
        )

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, self._get_info(reward)

    def _clip_position(self, position):
        return np.clip(position, 0.0, self.world_size).astype(np.float32)

    def _get_obs(self):
        rel = self.robot_position - self.human_position
        return build_rl_observation(
            rel,
            self.human_orientation,
            self.robot_orientation,
            self.follow_mode,
            self.desired_distance,
            obs_mode=self.obs_mode,
        )

    def _get_info(self, reward=0.0):
        diag = local_diagnostics(
            self.robot_position - self.human_position,
            self.human_orientation,
            self.robot_orientation,
            self.follow_mode,
            self.desired_distance,
        )
        return {
            "reward": float(reward),
            **diag,
        }
