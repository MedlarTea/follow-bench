import os
import warnings

import torch
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import DQN


def resolve_torch_device():
    requested = os.environ.get("FOLLOW_AHEAD_DEVICE", "").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        preferred = True
    else:
        preferred = torch.cuda.is_available()

    if preferred:
        try:
            probe = torch.zeros(1, device="cuda")
            _ = probe.cpu()
            return torch.device("cuda")
        except Exception as exc:
            warnings.warn(
                f"CUDA is available but unusable for Follow-Ahead-Reaction; "
                f"falling back to CPU. Details: {exc}"
            )

    return torch.device("cpu")


DEVICE = resolve_torch_device()


class RL_model:
    def __init__(self):
        return

    def load_model(self, model_path="", policy="a2c", env=None):
        if policy == 'dqn':
            if env is None:
                from nav_env import Environment

                env = Environment()
            self.model = DQN('MlpPolicy', env, verbose=1, buffer_size=10000, learning_rate=1e-3, batch_size=32, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.02)
            self.model.q_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        elif policy == 'a2c':
            self.model = A2C.load(model_path, device=DEVICE)
        elif policy == 'ddpg':
            self.model = DDPG.load(model_path, device=DEVICE)
        else:
            raise Exception
        return self.model    

    def evaluate_state(self, state, action=None, policy='a2c'):
        assert action is None or policy == 'dqn'
        assert policy is not None
        state = state.to(DEVICE)
        if policy == 'dqn':
            q_values = self.model.policy.q_net(state).detach()
            q_values = q_values.flatten()
            if action is None:
                return torch.max(q_values)
            return q_values[action]
        elif policy == 'a2c':
            value = self.model.policy.predict_values(state) 
            return value
        elif policy == 'ddpg':
            q_values = self.model.critic(state).detach()
            q_values = q_values.flatten()
            if action is None:
                return torch.max(q_values)
            return q_values[action]
        else:
            raise Exception()
