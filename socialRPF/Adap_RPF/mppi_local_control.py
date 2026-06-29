#!/usr/bin/env python

import time

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import torch
import pytorch_mppi as mppi


mppi_default_config = {
    "safe_dist": 0.8,
    "obstacle_cost_weight": 1.5,
    "goal_cost_weight": 0.5,
    "control_cost_weight": 0.05,
    "noise": 2.0,
    "samples": 250,
}


class MPPILocalController:
    def __init__(self, config=None, dt=0.1, prediction_horizon=2.0):
        mppi_config = config or mppi_default_config
        self.safe_dist = mppi_config.get("safe_dist", 0.8)
        self.obstacle_weight = mppi_config.get("obstacle_cost_weight", 1.5)
        self.goal_weight = mppi_config.get("goal_cost_weight", 0.5)
        self.control_weight = mppi_config.get("control_cost_weight", 0.05)

        self.mppi_noise = mppi_config.get("noise", 2.0)
        self.mppi_samples = mppi_config.get("samples", 250)
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mppi_settings = {
            "noise": torch.tensor([2.0, 0.8], device=self.device),
            "samples": 500,
            "horizon": int(self.prediction_horizon / self.dt),
        }

        self.predicted_traj = None
        self.goal_traj = None
        self.u_prev = None
        self.v_pref = 2.0
        self.step_counter = 0
        self.mppi_ctrl = self.create_mppi()

    def reset(self):
        self.predicted_traj = None
        self.goal_traj = None
        self.u_prev = None
        self.step_counter = 0
        self.mppi_ctrl = self.create_mppi()

    def dynamics(self, state, action, t=None):
        v = action[:, 0]
        omega = action[:, 1]
        theta = state[:, 2]

        next_state = torch.empty_like(state)
        next_state[:, 0] = state[:, 0] + v * torch.cos(theta) * self.dt
        next_state[:, 1] = state[:, 1] + v * torch.sin(theta) * self.dt
        next_state[:, 2] = state[:, 2] + omega * self.dt
        return next_state

    def cost(self, state, action, t):
        t_idx = int(t.item()) if torch.is_tensor(t) and t.numel() == 1 else int(t)
        if self.goal_traj is None:
            return torch.zeros_like(state[:, 0])

        cost_goal = self.goal_cost(state, t_idx)
        cost_obstacle = self.obstacle_cost(state, t_idx)
        cost_control = torch.norm(action, dim=1)

        total_cost = (
            self.goal_weight * cost_goal
            + self.obstacle_weight * cost_obstacle
            + self.control_weight * cost_control
        )
        return total_cost

    def goal_cost(self, state, t_idx):
        t_idx = min(t_idx, self.goal_traj.shape[0] - 1)
        target_point_on_path = self.goal_traj[t_idx]
        robot_positions = state[:, :2]
        return torch.sum((robot_positions - target_point_on_path) ** 2, dim=1)

    def obstacle_cost(self, state, t_idx):
        num_samples = state.shape[0]
        if self.predicted_traj is None or self.predicted_traj.shape[1] <= 1:
            return torch.zeros(num_samples, device=self.device)

        t_idx = min(t_idx, self.predicted_traj.shape[0] - 1)
        pred_obs = self.predicted_traj[t_idx, 1:, :]
        if pred_obs.shape[0] == 0:
            return torch.zeros(num_samples, device=self.device)

        obs_pos = torch.tensor(pred_obs[:, :2], device=self.device, dtype=torch.float32)
        obs_vel = torch.tensor(pred_obs[:, 2:4], device=self.device, dtype=torch.float32)
        robot_pos = state[:, :2]
        robot_theta = state[:, 2]

        robot_vel = torch.stack(
            [
                self.v_pref * 0.5 * torch.cos(robot_theta),
                self.v_pref * 0.5 * torch.sin(robot_theta),
            ],
            dim=1,
        )

        robot_pos_exp = robot_pos.unsqueeze(1)
        obs_pos_exp = obs_pos.unsqueeze(0)
        robot_vel_exp = robot_vel.unsqueeze(1)
        obs_vel_exp = obs_vel.unsqueeze(0)

        dists_sq = torch.sum((robot_pos_exp - obs_pos_exp) ** 2, dim=-1)
        encroachment_cost = torch.relu(self.safe_dist**2 - dists_sq)

        relative_pos = obs_pos_exp - robot_pos_exp
        relative_vel = robot_vel_exp - obs_vel_exp
        relative_vel_sq_norm = torch.sum(relative_vel**2, dim=-1)
        relative_vel_sq_norm = torch.clamp(relative_vel_sq_norm, min=1e-6)

        dot_prod = torch.sum(relative_pos * relative_vel, dim=-1)
        ttc = -dot_prod / relative_vel_sq_norm
        ttc = torch.clamp(ttc, min=0.0, max=20.0)

        dca_sq = torch.sum(relative_pos**2, dim=-1) + ttc * dot_prod
        dca_sq = torch.clamp(dca_sq, min=1e-4, max=10.0)

        k_ttc = 0.8
        k_dca = 2.0
        predictive_risk = torch.exp(-k_dca * dca_sq) * torch.exp(-k_ttc * ttc)
        predictive_risk = predictive_risk * (ttc > 0)

        w_encroach = 1.0
        w_predictive = 0.0
        total_cost = (
            w_encroach * torch.sum(encroachment_cost, dim=1)
            + w_predictive * torch.sum(predictive_risk, dim=1)
        )
        total_cost[torch.isnan(total_cost)] = 0.0
        total_cost[torch.isinf(total_cost)] = 1e3
        return torch.clamp(total_cost, min=0.0, max=1e5)

    def create_mppi(self):
        u_init = self.u_prev.unsqueeze(0) if self.u_prev is not None and self.u_prev.dim() == 1 else self.u_prev
        num_obstacles = (
            self.predicted_traj.shape[1] - 1
            if self.predicted_traj is not None and self.predicted_traj.shape[1] > 1
            else 0
        )
        adaptive_lambda = 1.0 + 0.1 * num_obstacles
        return mppi.MPPI(
            self.dynamics,
            self.cost,
            nx=3,
            noise_sigma=torch.diag(self.mppi_settings["noise"]),
            num_samples=self.mppi_settings["samples"],
            horizon=self.mppi_settings["horizon"],
            lambda_=adaptive_lambda,
            device=self.device,
            u_min=torch.tensor([0, -np.pi / 2], device=self.device, dtype=torch.float32),
            u_max=torch.tensor([self.v_pref, np.pi / 2], device=self.device, dtype=torch.float32),
            u_init=u_init,
            step_dependent_dynamics=True,
        )

    def control(self, robot_pose, goal_traj, predicted_traj=None):
        start_t = time.time()
        self.predicted_traj = predicted_traj
        self.goal_traj = torch.as_tensor(goal_traj, device=self.device, dtype=torch.float32)

        try:
            robot_state_np = np.asarray(robot_pose).squeeze()
            robot_state = torch.tensor(robot_state_np, device=self.device, dtype=torch.float32).unsqueeze(0)
            mppi_action = self.mppi_ctrl.command(robot_state)
            solve_time = time.time() - start_t
            self.step_counter += 1
            return mppi_action.cpu().numpy().flatten(), {
                "solve_time": solve_time,
                "success": True,
            }
        except Exception as exc:
            print(f"MPPI 控制器在计算时出错: {exc}")
            import traceback

            traceback.print_exc()
            self.step_counter += 1
            return np.zeros(2), {
                "solve_time": time.time() - start_t,
                "success": False,
                "error": str(exc),
            }

    def visualize_mppi_samples(self, env):
        mppi_controller = self.mppi_ctrl
        if self.predicted_traj is None or not hasattr(mppi_controller, "states"):
            return
        if (
            mppi_controller.states is None
            or getattr(mppi_controller, "cost_total", None) is None
            or getattr(mppi_controller, "state", None) is None
        ):
            return

        sample_trajectories_tensor = mppi_controller.states.cpu().squeeze(0)
        costs = mppi_controller.cost_total.cpu().numpy()
        nominal_trajectory_tensor = mppi_controller.get_rollouts(mppi_controller.state, num_rollouts=1)
        if nominal_trajectory_tensor is None:
            return
        nominal_trajectory = nominal_trajectory_tensor.cpu().squeeze().numpy()

        num_samples_to_draw = min(20, self.mppi_samples // 10)
        min_cost = costs.min()
        max_cost = costs.max()
        cost_range = max(max_cost - min_cost, 1e-6)
        cmap = cm.get_cmap("coolwarm_r")
        if num_samples_to_draw > sample_trajectories_tensor.shape[0]:
            num_samples_to_draw = sample_trajectories_tensor.shape[0]

        for i in range(num_samples_to_draw):
            traj_points = []
            for state in sample_trajectories_tensor[i]:
                x, y, theta = state[:3]
                traj_points.append(np.array([[x], [y], [theta]]))
            norm_cost = (costs[i] - min_cost) / cost_range
            color = cmap(norm_cost)
            hex_color = colors.rgb2hex(color)
            opacity = 0.2 + 0.6 * (1 - norm_cost)
            opacity = max(0.1, min(0.8, opacity))
            try:
                env.draw_trajectory(traj_points, color=hex_color, alpha=opacity, refresh=True)
            except TypeError:
                env.draw_trajectory(traj_points, traj_type="c-", refresh=True)

        nominal_points = []
        for state in nominal_trajectory:
            x, y, theta = state[:3]
            nominal_points.append(np.array([[x], [y], [theta]]))

        try:
            env.draw_trajectory(nominal_points, traj_type="b-", alpha=0.9, refresh=True)
        except TypeError:
            env.draw_trajectory(nominal_points, traj_type="b-", refresh=True)
