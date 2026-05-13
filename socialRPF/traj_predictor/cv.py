import numpy as np               
import matplotlib.pyplot as plt                                                                                               
class CV(object):
    def __init__(self):
        super(CV, self).__init__()
        self.dt = None
        self.prediction_horizon = None
        self.wrap = np.vectorize(self._wrap)
    
    def set_params(self, params):
        self.dt = params['dt']
        self.prediction_horizon = params['prediction_horizon']
        self.history_length = params['history_length']
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1

    
    def reset(self):
        pass

    def get_predictions(self, trajectory):
        velocity = trajectory[None, -1, 1:, 2:4]  # 1 x H x 2
        init_pos = trajectory[None, -1, 1:, 0:2]  # 1 x H x 2

        steps = 1 + np.arange(self.prediction_length, dtype=np.float64)[:, None, None]  # T' x 1 x 1
 #       rospy.loginfo("1: {}".format(steps.shape))
        steps = np.multiply(velocity, steps) * self.dt  # T' x H x 2
        steps = (init_pos + steps)[None, None] # N x S x T' x H x 2
        steps = np.concatenate((steps[:, :, :-1], self.predict_velocity(steps)), axis=-1) # N x S x T' x H x 4
#        rospy.loginfo("{} {}\n\n".format(steps.shape, self.prediction_length))
        return steps
    
    def predict_velocity(self, steps):
        return (steps[:, :, 1:]-steps[:, :, :-1])/self.dt

    def predict(self, trajectory): # (T x (1+H) x 5), ((1+H) x 5), (N x T' x 2), (2, )
        predictions = self.get_predictions(trajectory) # N x S x T' x H x 4
        return predictions

    def predictor_cost(self, state, actions, predictions):
        return np.array([0.0])
    
    def discrete_cost(self, state, actions, predictions): # (1+H) x 5, N x T' x 4, N x S x T' x H x 4
        N = actions.shape[0]
        S = predictions.shape[1]
        state_ = np.tile(state[None, None, None, None, 0, :2]-state[None, None, None, 1:, :2], (N, S, 1, 1, 1))
        dxdy = np.concatenate((state_, actions[:, None, :, None, :2] - predictions[:, :, :, :, :2]), axis=2)
        winding_nums = np.arctan2(dxdy[:, :, :, :, 1], dxdy[:, :, :, :, 0]) # N x S x T' x H
        winding_nums = winding_nums[:, :, 1:]-winding_nums[:, :, :-1]

        if self.discrete_cost_type == 'entropy':
            winding_nums = np.mean(winding_nums, axis=2) < 0 # N x S x H
            p = np.mean(winding_nums, axis=1) # N x H
            # Using mean entropy
            entropy = - (p * np.log(p+1e-8) + (1-p) * np.log(1-p+1e-8))
            entropy = np.mean(entropy, axis=1)[:, None]

            return self.Q_discrete * (entropy ** 2)
        else:
            winding_nums = np.abs(np.mean(winding_nums, axis=2)) # N x S x H

            # considering all agents we are in front of
            dxdy = state[None, 0, :2] - state[1:, :2]
            obs_theta = np.arctan2(state[1:, 3], state[1:, 2])
            alpha = self.wrap(np.arctan2(dxdy[:, 1], dxdy[:, 0]) - obs_theta + np.pi/2.0) >= 0 # N x S x H
            winding_nums = np.multiply(winding_nums, alpha)
            
            winding_nums = np.multiply(winding_nums, alpha)
            winding_nums = np.mean(winding_nums, axis=-1) # N x S

            return - self.Q_discrete * (winding_nums ** 2)
        

    @staticmethod
    def _wrap(angle):  # keep angle between [-pi, pi]
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

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
    
    # stack trajectory and velocity into shape (N, T, 4)
    trajectory = np.array(trajectory)
    velocities = np.array(velocities)
    
    return np.concatenate([trajectory, velocities], axis=1)

if __name__ == '__main__':
    traj_predictor = CV()
    param = {
        "dt": 0.1,
        "prediction_horizon": 2.0

    }
    traj_predictor.set_params(params=param)
    # generate constant-velocity straight-line trajectories for 4 targets
    N = 4  # 4 people
    T = 8  # each trajectory contains 8 points

    # assume each person has a different start position, direction and speed
    start_positions = [(0, 0), (3, 4), (6, 4), (4.5, 6)]
    speeds = [1.0, 1.0, 1.0, 1.0]  # everyone moves at the same speed
    directions = [np.pi/4, 0.0, np.pi, -np.pi/2]  # different heading per person

    trajectories_with_velocity = np.zeros((T, N, 4))

    for i in range(N):
        trajectories_with_velocity[:, i, :] = generate_straight_line_trajectory(T, start_positions[i], speeds[i], param["dt"], directions[i])
    
     # reshape to (N, T, 4)
    trajectories_with_velocity = np.array(trajectories_with_velocity)
    print("history shape:", trajectories_with_velocity.shape)
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