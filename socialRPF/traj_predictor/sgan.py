"""
Based on https://github.com/fluentrobotics/ComplexityNav
"""

import os
from traj_predictor.cv import CV
import numpy as np
import torch
from traj_predictor.utils.optimized_sgan import OptimizedSGAN
from traj_predictor.utils.sgan_utils import get_generator, optimized_relative_to_abs
import matplotlib.pyplot as plt    
# import rospy

class SGAN(CV):
    def __init__(self):
        super(SGAN, self).__init__()
        self.model = OptimizedSGAN

    def set_params(self, params):
        self.dt = params['dt']
        self.prediction_horizon = params['prediction_horizon']
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(
            np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = params['history_length']
        # self.log_cost = params['log_cost']

        # self.q_obs = params['cost']['q']['obs']
        # self.q_goal = params['cost']['q']['goal']
        # self.q_wind = params['cost']['q']['wind']
        # self.q_dev = params['cost']['q']['dev']

        # self.sigma_h = params['cost']['sigma']['h']
        # self.sigma_s = params['cost']['sigma']['s']
        # self.sigma_r = params['cost']['sigma']['r']

        # Normalization factors for Q weights
        self.q_goal_norm = np.square(2 / float(self.prediction_horizon))

        # # Empirically found
        # q_wind_norm = 0.1 * np.deg2rad(350)
        # self.q_wind_norm = np.square(q_wind_norm)
        # self.q_obs_norm = np.square(0.5)

        # Normalized weights
        # self.Q_obs = self.q_obs / self.q_obs_norm
        # self.Q_goal = self.q_goal / self.q_goal_norm
        # self.Q_discrete = self.q_wind / self.q_wind_norm
        # self.Q_dev = self.q_dev
        # self.discrete_cost_type = params['cost']['discrete_cost_type']

        self.iskip = 4
        self.oskip = int(np.ceil(0.4 / self.dt))
        self.model_obs_len = self.history_length
        self.model_pred_len = int(np.ceil(self.prediction_horizon / 0.4))+1

        self.device = torch.device(
            "cuda" if params['predictor']['use_gpu'] else "cpu")
        self.num_samples = params['predictor']['num_samples']
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", params['predictor']['path'])
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint['args']['pred_len'] = self.model_pred_len
        checkpoint['args']['obs_len'] = self.model_obs_len
        checkpoint['args']['batch_size'] = 1
        self.generator = get_generator(self.model, checkpoint, self.device)

        self.deviation_penalty = params['predictor']['deviation_penalty']
        if params['predictor']['use_sgan_action']:
            self.predict = self._predict
        
        self.use_mode = params['predictor']['use_sgan_mode']

    def create_input(self, trajectory):
        indexes = torch.flip(torch.arange(
            len(trajectory)-1, -1, -self.iskip, device=self.device)[:self.history_length+1], dims=(0,))
        trajectory = trajectory[indexes, :, :2] # (T'+1) x (1+H) x 2
        if len(trajectory) <= 1:
            trajectory = torch.nn.functional.pad(trajectory, (0, 0, 0, 0, 1, 0), mode='constant')
        obs_traj = trajectory[1:] # T' x (1+H) x 2
        obs_traj_rel = obs_traj - trajectory[0:-1] # T' x (1+H) x 2
                                                                                                                # rospy.loginfo("Input: {}".format(obs_traj.shape, obs_traj_rel.shape))
        return obs_traj, obs_traj_rel # T' x (1+H) x 2
        
    def create_output(self, traj, init_pos):  # N x S x T'' x H x 2, H x 5
        N = traj.shape[0]
        S = traj.shape[1]
        H = traj.shape[-2]
        init_pos = init_pos[None, None, None, :, :2].expand(N, S, -1, -1, -1)
        traj = torch.cat((init_pos, traj), axis=2)

        #TODO: optimize
        new_traj = []
        for t in range(traj.shape[2]-1):
            for i in range(self.oskip):
                new_traj.append(
                    (traj[:, :, t]*(self.oskip-i)+traj[:, :, t+1]*i)/self.oskip)
        new_traj.append(traj[:, :, -1])
        new_traj = torch.stack(new_traj, axis=2)
        # N x S x T' x H x 2
        return new_traj[:, :, 1:self.prediction_length+1]

    def get_predictions(self, trajectory):
        with torch.no_grad():
            trajectory = torch.tensor(trajectory, device=self.device, dtype=torch.float)
            obs_traj, obs_traj_rel = self.create_input(trajectory)
            seq_start_end = torch.tensor(
                [[0, obs_traj.shape[1]]], dtype=torch.int, device=self.device)
            noise = None
            if self.use_mode:
                noise = torch.ones((obs_traj.shape[0],)+self.generator.noise_dim, device=self.device)
            pred_traj_fake_rel = self.generator(
                obs_traj, obs_traj_rel, seq_start_end, self.num_samples, noise)
            pred_traj_fake = optimized_relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1])[None] # N x S x T'' x (1+H) x 2
                                                                                                                    # rospy.loginfo("from model: {}".format(pred_traj_fake.shape))
            pred_traj_fake = self.create_output(
                pred_traj_fake[:, :, :], trajectory[-1])  # N x S x T' x (1+H) x 2
            
            pred_traj_fake = torch.cat(
                (pred_traj_fake[:, :, :-1], self.predict_velocity(pred_traj_fake)), dim=-1)  # N x S x T' x H x 4

            self.ego_traj_fake = pred_traj_fake[:, :, :, 0].cpu().numpy() # N x S x T' x 2
            pred_traj_fake = pred_traj_fake[:, :, :, 1:]

                                                                                                                    # rospy.loginfo("final_pred: {}\n\n".format(pred_traj_fake.shape))
        return pred_traj_fake.cpu().numpy()  # N x S x T' x H x 4
    
    def predictor_cost(self, state, actions, predictions):
        cost = np.array([0.0])
        if self.deviation_penalty:
            cost = np.mean(np.linalg.norm(actions[:, None, :, :2] - self.ego_traj_fake[:, :, :, :2], axis=-1), axis=-1)
        return self.Q_dev * (cost**2)
    
    def predict(self, trajectory): # (T x (1+H) x 5), ((1+H) x 5), (N x T' x 2), (2, )
        predictions = self.get_predictions(trajectory)[0,:, None] # N x S x T' x H x 4
        
        return predictions

def generate_straight_line_trajectory(steps, start_position=(0, 0), speed=1.0, dt=0.2, direction=np.pi/4):
    # constant per-step speed along the given direction
    x, y = start_position
    trajectory = [(x, y)]
    velocities = [(speed * np.cos(direction), speed * np.sin(direction))]  # initial velocity
    
    for _ in range(steps - 1):
        # update the position
        x += speed * np.cos(direction) * dt
        y += speed * np.sin(direction) * dt
        trajectory.append((x, y))
        
        # the velocity is constant, so every step has the same velocity
        velocities.append((speed * np.cos(direction), speed * np.sin(direction)))
    
    # stack trajectory and velocity into shape (N, T, 4)
    trajectory = np.array(trajectory)
    velocities = np.array(velocities)
    
    return np.concatenate([trajectory, velocities], axis=1)

if __name__ == '__main__':
    param = {
        "dt": 0.1,
        "prediction_horizon": 2.0,
        "history_length": 8,
        "predictor": {
            "path": 'sgan.pt',
            "use_gpu": False,
            "num_samples": 1,
            "deviation_penalty": True,
            "use_sgan_action": False,
            "use_sgan_mode": True,
        }
    }
    traj_predictor = SGAN()
    traj_predictor.set_params(params=param)
    # generate constant-velocity straight-line trajectories for 4 targets
    N = 4  # 4 agents (a robot and three pedestrians)
    T = param['history_length']  # number of points per trajectory

    # assume each person has a different start position, direction and speed
    start_positions = [(0, 0), (3, 4), (6, 4), (4.5, 6)]
    speeds = [1.0, 1.0, 1.0, 1.0]  # everyone moves at the same speed
    directions = [np.pi/4, 0.0, np.pi, -np.pi/2]  # different heading per person

    trajectories_with_velocity = np.zeros((T, N, 4))

    for i in range(N):
        trajectories_with_velocity[:, i, :] = generate_straight_line_trajectory(T, start_positions[i], speeds[i], param["dt"], directions[i])
    
     # reshape to (N, T, 4)
    trajectories_with_velocity = np.array(trajectories_with_velocity)
    print("history shape:", trajectories_with_velocity[:, 1:, :].shape)
    predictions = traj_predictor.predict(trajectories_with_velocity)
    print("Predictions shape:", predictions.shape)  # print the predicted shape
    predictions = predictions.squeeze()

    # plot the trajectories of all targets
    plt.figure(figsize=(10, 6))

    # iterate over each target and plot the history vs. future
    for i in range(predictions.shape[1]):
        # extract history and future trajectory for the current target
        history = trajectories_with_velocity[:, i+1, :2]  # take only positions (x, y)
        future = predictions[:, i, :2]  # take only positions (x, y)
        
        # plot the history trajectory (light color = past)
        plt.plot(history[:, 0], history[:, 1], 'b-', alpha=0.6, label=f'Target {i+1} History' if i == 0 else "")
        
        # plot the future trajectory (dark color = future)
        plt.plot(future[:, 0], future[:, 1], 'r-', alpha=0.8, label=f'Target {i+1} Future' if i == 0 else "")

    # axis labels and legend
    plt.title("Historical and Future Trajectories of Targets")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper left')

    # show the figure
    plt.grid(True)
    plt.show()