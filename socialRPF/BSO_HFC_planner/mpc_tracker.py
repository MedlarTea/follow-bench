from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from .bspline_opt import BSplineResult


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    wrapped = np.arctan2(np.sin(angle), np.cos(angle))
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class MPCConfig:
    """MPC parameters in yaml order, with timestep and hard bounds injected last."""

    horizon: int
    max_iter: int
    warm_start: bool
    allow_reverse: bool
    q_pos: float
    q_yaw: float
    q_v: float
    q_omega: float
    qf_pos: float
    qf_yaw: float
    qf_v: float
    qf_omega: float
    r_acc_v: float
    r_acc_omega: float
    rd_acc_v: float
    rd_acc_omega: float
    dt: float
    min_v: float
    max_v: float
    min_omega: float
    max_omega: float
    max_acc_v: float
    max_acc_omega: float


@dataclass
class TrackingReference:
    states: np.ndarray
    inputs: np.ndarray


class MPCTracker:
    """
    Linear reference-tracking MPC for the time-parameterized B-spline reference.
    """

    def __init__(self, cfg: MPCConfig) -> None:
        self.cfg = cfg
        self.prev_input = np.zeros((2,), dtype=float)
        self.prev_solution_u: np.ndarray | None = None

        self.q_pos = float(cfg.q_pos)
        self.qf_pos = float(cfg.qf_pos)
        self.q_state = np.array([cfg.q_yaw, cfg.q_v, cfg.q_omega], dtype=float)
        self.qf_state = np.array([cfg.qf_yaw, cfg.qf_v, cfg.qf_omega], dtype=float)
        self.r = np.array([cfg.r_acc_v, cfg.r_acc_omega], dtype=float)
        self.rd = np.array([cfg.rd_acc_v, cfg.rd_acc_omega], dtype=float)

    def reset(self) -> None:
        self.prev_input = np.zeros((2,), dtype=float)
        self.prev_solution_u = None

    def control(self, robot_vel: np.ndarray, spline_result: BSplineResult) -> tuple[np.ndarray, np.ndarray]:
        reference = self._build_reference(spline_result)
        if reference is None or len(reference.states) < 2:
            return np.zeros((2, 1), dtype=float), np.zeros((1, 2), dtype=float)

        horizon = int(max(2, self.cfg.horizon))
        x_ref = self._pad_reference_states(reference.states, horizon + 1)
        u_ref = self._pad_reference_inputs(reference.inputs, horizon)

        vel = np.asarray(robot_vel, dtype=float).reshape(-1)
        x0 = np.array([0.0, 0.0, 0.0, float(vel[0]), float(vel[1])], dtype=float)
        optimal_sequence = self._solve_mpc(x0, x_ref, u_ref)
        if optimal_sequence is None:
            optimal_sequence = u_ref.copy()

        first_input = optimal_sequence[0]
        self.prev_input = np.asarray(first_input, dtype=float)
        v_cmd = clamp(x0[3] + self.prev_input[0] * self.cfg.dt, self.cfg.min_v, self.cfg.max_v)
        omega_cmd = clamp(x0[4] + self.prev_input[1] * self.cfg.dt, self.cfg.min_omega, self.cfg.max_omega)
        pred_path = self._rollout_nonlinear(x0, optimal_sequence)
        return np.array([[v_cmd], [omega_cmd]], dtype=float), pred_path

    def _build_reference(self, spline_result: BSplineResult | None) -> TrackingReference | None:
        """Turn the current remaining B-spline into a time-aligned MPC reference."""
        if spline_result is None or len(spline_result.samples) == 0:
            return None

        raw_samples = np.asarray(spline_result.samples, dtype=float)
        raw_headings = np.unwrap(np.asarray(spline_result.headings, dtype=float))
        raw_times = np.asarray(spline_result.sample_times, dtype=float).reshape(-1)
        samples, headings, sample_times = self._trim_reference_to_remaining_path(raw_samples, raw_headings, raw_times)
        num_samples = len(samples)
        if num_samples == 0:
            return None

        if num_samples == 1:
            samples = np.repeat(samples, 2, axis=0)
            headings = np.repeat(headings, 2)
            dt_seed = max(float(spline_result.delta_t), self.cfg.dt, 1e-3)
            sample_times = np.array([0.0, dt_seed], dtype=float)
        else:
            sample_times = np.asarray(sample_times, dtype=float)
            sample_times = sample_times - sample_times[0]
            if sample_times[-1] <= 1e-6:
                dt_seed = max(float(spline_result.delta_t), self.cfg.dt, 1e-3)
                sample_times = dt_seed * np.arange(num_samples, dtype=float)

        horizon_time = max(sample_times[-1], self.cfg.horizon * self.cfg.dt)
        total_steps = max(int(np.ceil(horizon_time / max(self.cfg.dt, 1e-6))), self.cfg.horizon) + 1
        times = np.linspace(0.0, (total_steps - 1) * self.cfg.dt, total_steps)

        x = np.interp(times, sample_times, samples[:, 0])
        y = np.interp(times, sample_times, samples[:, 1])
        yaw = np.interp(times, sample_times, headings)
        pos = np.column_stack((x, y))

        dt = max(self.cfg.dt, 1e-6)
        vel_xy = np.zeros_like(pos)
        vel_xy[:-1] = np.diff(pos, axis=0) / dt
        vel_xy[-1] = vel_xy[-2]
        speed = np.linalg.norm(vel_xy, axis=1)

        yaw_delta = wrap_angle(np.diff(yaw)) / dt if len(yaw) > 1 else np.zeros((0,), dtype=float)
        omega = np.zeros((len(yaw),), dtype=float)
        if yaw_delta.size:
            omega[:-1] = yaw_delta
            omega[-1] = yaw_delta[-1]

        acc_v = np.zeros((len(speed),), dtype=float)
        acc_omega = np.zeros((len(omega),), dtype=float)
        if len(speed) > 1:
            acc_v[:-1] = np.diff(speed) / dt
            acc_omega[:-1] = np.diff(omega) / dt

        states = np.column_stack((pos[:, 0], pos[:, 1], yaw, speed, omega))
        inputs = np.column_stack((acc_v, acc_omega))
        return TrackingReference(states=states, inputs=inputs)

    def _pad_reference_states(self, states: np.ndarray, target_len: int) -> np.ndarray:
        if len(states) >= target_len:
            return np.asarray(states[:target_len], dtype=float)
        pad = np.repeat(states[-1:], target_len - len(states), axis=0)
        return np.vstack((states, pad))

    def _pad_reference_inputs(self, inputs: np.ndarray, target_len: int) -> np.ndarray:
        if len(inputs) == 0:
            return np.zeros((target_len, 2), dtype=float)
        if len(inputs) >= target_len:
            return np.asarray(inputs[:target_len], dtype=float)
        pad = np.repeat(inputs[-1:], target_len - len(inputs), axis=0)
        return np.vstack((inputs, pad))

    def _trim_reference_to_remaining_path(
        self,
        samples: np.ndarray,
        headings: np.ndarray,
        sample_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trim the reused spline so MPC tracks only the forward remaining path."""
        path = np.asarray(samples, dtype=float)
        yaw = np.asarray(headings, dtype=float)
        times = np.asarray(sample_times, dtype=float)
        if len(path) == 0:
            return path, yaw, times

        start_idx = self._select_progress_start_index(path)
        path = path[start_idx:]
        yaw = yaw[start_idx:]
        times = times[start_idx:]
        if len(path) == 0:
            return path, yaw, times

        if np.linalg.norm(path[0]) > 1e-6:
            path = np.vstack((np.zeros((1, 2), dtype=float), path))
            yaw = np.concatenate((yaw[:1], yaw))
            start_time = float(times[0]) if len(times) > 0 else 0.0
            times = np.concatenate(([start_time], times))
        else:
            path = path.copy()
            path[0] = 0.0

        if len(times) == 0:
            times = np.zeros((len(path),), dtype=float)
        elif len(times) != len(path):
            first_time = float(times[0])
            times = np.concatenate(([first_time], times))
        return path, yaw, times

    def _select_progress_start_index(self, samples: np.ndarray) -> int:
        """Pick the earliest nearby forward sample to avoid jumping ahead across a bend."""
        path = np.asarray(samples, dtype=float)
        if len(path) <= 1:
            return 0

        distances = np.linalg.norm(path, axis=1)
        behind_margin = max(float(self.cfg.max_v) * float(self.cfg.dt), 0.05)
        forward_mask = path[:, 0] >= -behind_margin
        if not np.any(forward_mask):
            return int(np.argmin(distances))

        candidate_indices = np.flatnonzero(forward_mask)
        candidate_distances = distances[candidate_indices]
        min_distance = float(np.min(candidate_distances))
        tie_margin = max(0.05, 0.5 * behind_margin)
        near_indices = candidate_indices[candidate_distances <= min_distance + tie_margin]
        return int(near_indices[0])

    @staticmethod
    def _align_angle(target: float, reference: float) -> float:
        return reference + wrap_angle(target - reference)

    def _linearized_step(self, ref_state: np.ndarray, ref_input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        px, py, yaw, v, omega = ref_state
        acc_v, acc_omega = ref_input
        dt = self.cfg.dt

        v_next = v + acc_v * dt
        omega_next = omega + acc_omega * dt
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        a = np.eye(5, dtype=float)
        a[0, 2] = -v_next * dt * sin_yaw
        a[0, 3] = dt * cos_yaw
        a[1, 2] = v_next * dt * cos_yaw
        a[1, 3] = dt * sin_yaw
        a[2, 4] = dt

        b = np.zeros((5, 2), dtype=float)
        b[0, 0] = dt * dt * cos_yaw
        b[1, 0] = dt * dt * sin_yaw
        b[2, 1] = dt * dt
        b[3, 0] = dt
        b[4, 1] = dt

        f = np.array(
            [
                px + v_next * dt * cos_yaw,
                py + v_next * dt * sin_yaw,
                yaw + omega_next * dt,
                v_next,
                omega_next,
            ],
            dtype=float,
        )
        c = f - a @ ref_state - b @ ref_input
        return a, b, c

    def _warm_start_guess(self, horizon: int, u_ref: np.ndarray) -> np.ndarray:
        if self.cfg.warm_start and self.prev_solution_u is not None and self.prev_solution_u.shape == (horizon, 2):
            return np.vstack((self.prev_solution_u[1:], self.prev_solution_u[-1:]))
        return np.asarray(u_ref[:horizon], dtype=float)

    def _build_condensed_qp(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
    ) -> tuple[sparse.csc_matrix, np.ndarray, sparse.csc_matrix, np.ndarray, np.ndarray, int, int]:
        """Condense the linearized tracking problem so OSQP only optimizes the input sequence."""
        horizon = int(max(2, self.cfg.horizon))
        nx = 5
        nu = 2
        total_u = horizon * nu

        x0_aligned = np.asarray(x0, dtype=float).copy()
        x0_aligned[2] = self._align_angle(x0_aligned[2], x_ref[0, 2])

        state_maps = np.zeros((horizon, nx, total_u), dtype=float)
        state_offsets = np.zeros((horizon, nx), dtype=float)
        g_prev = np.zeros((nx, total_u), dtype=float)
        d_prev = x0_aligned

        for k in range(horizon):
            a, b, c = self._linearized_step(x_ref[k], u_ref[k])
            g_next = a @ g_prev
            col = k * nu
            g_next[:, col : col + nu] += b
            d_next = a @ d_prev + c
            state_maps[k] = g_next
            state_offsets[k] = d_next
            g_prev = g_next
            d_prev = d_next

        target_states = np.asarray(x_ref[1 : horizon + 1], dtype=float).copy()
        target_states[:, 2] = np.unwrap(target_states[:, 2])
        state_rows, state_rhs = self._build_tracking_rows(state_maps, state_offsets, target_states)

        sqrt_r = np.sqrt(self.r)
        a_u = np.kron(np.eye(horizon), np.diag(sqrt_r))
        b_u = (u_ref[:horizon] * sqrt_r).reshape(-1)

        diff_matrix = np.eye(total_u, dtype=float)
        eye_u = np.eye(nu, dtype=float)
        for k in range(1, horizon):
            row = k * nu
            diff_matrix[row : row + nu, row - nu : row] = -eye_u
        sqrt_rd = np.kron(np.eye(horizon), np.diag(np.sqrt(self.rd)))
        a_du = sqrt_rd @ diff_matrix
        diff_target = np.zeros((horizon, nu), dtype=float)
        diff_target[0] = self.prev_input
        b_du = (sqrt_rd @ diff_target.reshape(-1, 1)).reshape(-1)

        a_ls = np.vstack((state_rows, a_u, a_du))
        b_ls = np.concatenate((state_rhs, b_u, b_du))

        p_dense = 2.0 * (a_ls.T @ a_ls)
        p_dense = 0.5 * (p_dense + p_dense.T) + 1e-9 * np.eye(total_u, dtype=float)
        q = -2.0 * (a_ls.T @ b_ls)

        lower = np.tile(np.array([-self.cfg.max_acc_v, -self.cfg.max_acc_omega], dtype=float), horizon)
        upper = np.tile(np.array([self.cfg.max_acc_v, self.cfg.max_acc_omega], dtype=float), horizon)
        a_box = sparse.eye(total_u, format='csc')
        return sparse.csc_matrix(p_dense), np.asarray(q, dtype=float), a_box, lower, upper, horizon, nu

    def _build_tracking_rows(
        self,
        state_maps: np.ndarray,
        state_offsets: np.ndarray,
        target_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build least-squares rows for reference-state tracking over the MPC horizon."""
        total_u = state_maps.shape[2]
        rows: list[np.ndarray] = []
        rhs: list[float] = []

        for k in range(len(target_states)):
            target_state = np.asarray(target_states[k], dtype=float)
            state_map = np.asarray(state_maps[k], dtype=float)
            state_offset = np.asarray(state_offsets[k], dtype=float)
            pos_weight = self.qf_pos if k == len(target_states) - 1 else self.q_pos
            state_weights = self.qf_state if k == len(target_states) - 1 else self.q_state

            self._append_weighted_state_row(
                rows,
                rhs,
                np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
                state_map,
                state_offset,
                target_state,
                pos_weight,
            )
            self._append_weighted_state_row(
                rows,
                rhs,
                np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
                state_map,
                state_offset,
                target_state,
                pos_weight,
            )
            self._append_weighted_state_row(
                rows,
                rhs,
                np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float),
                state_map,
                state_offset,
                target_state,
                state_weights[0],
            )
            self._append_weighted_state_row(
                rows,
                rhs,
                np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float),
                state_map,
                state_offset,
                target_state,
                state_weights[1],
            )
            self._append_weighted_state_row(
                rows,
                rhs,
                np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float),
                state_map,
                state_offset,
                target_state,
                state_weights[2],
            )

        if not rows:
            return np.zeros((0, total_u), dtype=float), np.zeros((0,), dtype=float)
        return np.vstack(rows), np.asarray(rhs, dtype=float)

    @staticmethod
    def _append_weighted_state_row(
        rows: list[np.ndarray],
        rhs: list[float],
        selector: np.ndarray,
        state_map: np.ndarray,
        state_offset: np.ndarray,
        target_state: np.ndarray,
        weight: float,
    ) -> None:
        if weight <= 0.0:
            return

        selector_row = np.asarray(selector, dtype=float).reshape(1, -1)
        weighted_row = np.sqrt(weight) * (selector_row @ state_map)
        weighted_rhs = np.sqrt(weight) * (selector_row @ (target_state - state_offset))
        rows.append(weighted_row.reshape(-1))
        rhs.append(float(weighted_rhs.reshape(-1)[0]))

    def _solve_mpc(self, x0: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray) -> np.ndarray | None:
        """Solve the boxed-input tracking QP with OSQP and return the future acceleration sequence."""
        p_mat, q_vec, a_box, lower, upper, horizon, nu = self._build_condensed_qp(x0, x_ref, u_ref)
        solver = osqp.OSQP()
        solver.setup(
            P=p_mat,
            q=q_vec,
            A=a_box,
            l=np.asarray(lower, dtype=float),
            u=np.asarray(upper, dtype=float),
            verbose=False,
            warm_start=bool(self.cfg.warm_start),
            polish=False,
            max_iter=int(self.cfg.max_iter),
            eps_abs=1e-4,
            eps_rel=1e-4,
        )

        if self.cfg.warm_start:
            init = self._warm_start_guess(horizon, u_ref).reshape(-1)
            solver.warm_start(x=init)

        result = solver.solve()
        status = str(getattr(result.info, 'status', '')).lower()
        if result.x is None or 'solved' not in status:
            return None

        self.prev_solution_u = np.asarray(result.x, dtype=float).reshape(horizon, nu)
        return self.prev_solution_u

    def _rollout_nonlinear(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        state = np.asarray(x0, dtype=float).copy()
        path = [state[:2].copy()]
        for u in np.asarray(u_seq, dtype=float):
            v_next = clamp(state[3] + float(u[0]) * self.cfg.dt, self.cfg.min_v, self.cfg.max_v)
            omega_next = clamp(state[4] + float(u[1]) * self.cfg.dt, self.cfg.min_omega, self.cfg.max_omega)
            state[0] += v_next * self.cfg.dt * np.cos(state[2])
            state[1] += v_next * self.cfg.dt * np.sin(state[2])
            state[2] = wrap_angle(state[2] + omega_next * self.cfg.dt)
            state[3] = v_next
            state[4] = omega_next
            path.append(state[:2].copy())
        return np.asarray(path, dtype=float)
