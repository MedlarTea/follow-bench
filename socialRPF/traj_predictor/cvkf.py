"""
Based on https://github.com/fluentrobotics/ComplexityNav
"""
# import rospy
import numpy as np
import copy
# import rospy
from traj_predictor.cv import CV
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt


class CVKF(CV):
    def __init__(self):
        super(CVKF, self).__init__()
        self.dt = None
        self.prediction_horizon = None
        self.filters = None
        self.wrap = np.vectorize(self._wrap)

    def set_params(self, params):
        self.dt = params['dt']
        self.prediction_horizon = params['prediction_horizon']
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = params['history_length']

        self.num_samples = params['predictor']['num_samples']

    def get_kf(self):
        kf = KalmanFilter(dim_x=4, dim_z=4)  # (x,y,vx,vy)
        kf.x = np.zeros(4)
        # constant-velocity model
        kf.F = np.array([[1., 0., self.dt, 0.], # x   = x0 + dx*dt
                        [0., 1., 0., self.dt],  # y   = y0 + dy*dt
                        [0., 0., 1., 0.],  # dx  = dx0
                        [0., 0., 0., 1.]])      # dy  = dy0
        kf.H = np.eye(4)  # Measurement matrix

        # Initial state covariance matrix
        pos_sigma0, vel_sigma0 = 0.1, 0.1         # m, m/s
        kf.P = np.diag([pos_sigma0**2, pos_sigma0**2,
                        vel_sigma0**2, vel_sigma0**2])

        # measurement noise covariance matrix
        pos_sigma_meas, vel_sigma_meas = 0.5, 0.5
        kf.R = np.diag([pos_sigma_meas**2, pos_sigma_meas**2,
                        vel_sigma_meas**2, vel_sigma_meas**2])

        # process noise covariance matrix
        # accel_sigma = 0.01                         # m/s²
        # G = np.array([[0.5*self.dt**2], [0.5*self.dt**2], [self.dt], [self.dt]])
        # kf.Q = G @ G.T * accel_sigma**2           # 4×4

        process_sigma = 0.05
        kf.Q = np.diag([process_sigma**2, process_sigma**2, 
                        process_sigma**2, process_sigma**2])  # Process noise covariance matrix
        return kf


    def reset(self):
        self.filters = None

    def unroll_kf(self, kf, s):
        kf_ = copy.deepcopy(kf)
        trajectory = []
        for _ in range(self.rollout_steps):
            kf_.predict()
            # rospy.loginfo("{} {} {}\n\n\n\n\n".format(kf_.x.shape, np.linalg.cholesky(kf_.P).shape, (self.num_samples, 4)))
            # trajectory.append(np.random.multivariate_normal(kf_.x.squeeze(), kf_.P, size=self.num_samples))
            trajectory.append(kf_.x)
        trajectory = np.stack(trajectory)
        return trajectory

    def get_predictions(self, trajectory):
        if self.filters is None:
            self.filters = [self.get_kf()
                            for _ in range(trajectory.shape[1]-1)]
            for kf, s in zip(self.filters, trajectory[-1, 1:, :4]):
                kf.x = s
        predictions = []
        for kf, s in zip(self.filters, trajectory[-1, 1:, :4]):
            # print(kf.x)
            kf.predict()
            # print(s)
            kf.update(s)

            nis = kf.y.T @ np.linalg.inv(kf.S) @ kf.y  # Normalized Innovation Squared
            nis = nis.repeat(self.rollout_steps)[:, np.newaxis]


            predicted_state = self.unroll_kf(kf, s)  # T' x 4
            predicted_state = np.hstack([predicted_state, nis])  # T' x 5
            # print(kf.x)
            # print("\nnis: {}\n".format(predicted_state[-1, 4]))
            predictions.append(predicted_state) # H x (T' x 5)
        
        # print(predictions[0])
        
        predictions = np.stack(predictions, axis=1)[np.newaxis, np.newaxis, :] # N x S x T' x H x 4
        return predictions
    
    def predict(self, trajectory): # (T x (1+H) x 5), ((1+H) x 5), (N x T' x 2), (2, )
        predictions = self.get_predictions(trajectory) # N x S x T' x H x 5
        return predictions

def generate_straight_line_trajectory(steps, start_position=(0, 0), speed=1.0, dt=0.1, direction=np.pi/4):
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
    
    # stack trajectory and velocity into shape (T, 4)
    trajectory = np.array(trajectory)
    velocities = np.array(velocities)
    
    return np.concatenate([trajectory, velocities], axis=1)

if __name__ == '__main__':
    param = {
        "dt": 0.25,
        "prediction_horizon": 1.25,
        "history_length": 1,
        "predictor": {
            "num_samples": 1,
        }
    }
    traj_predictor = CVKF()
    traj_predictor.set_params(params=param)
    # generate constant-velocity straight-line trajectories for 4 targets
    N = 4  # 4 agents (a robot and three pedestrians)
    T = param['history_length']  # number of points per trajectory
    
    # assume each person has a different start position, direction and speed
    start_positions = [(0, 0), (4, 4), (0, 5), (5, 5)]
    speeds = [1.0, 1.0, 1.0, 1.0]  # everyone moves at the same speed
    directions = [np.pi/4, np.pi/6, np.pi/3, np.pi/2]  # different heading per person

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
