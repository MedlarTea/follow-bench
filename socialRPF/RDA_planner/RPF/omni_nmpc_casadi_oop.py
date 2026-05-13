# MIT License

# Copyright (c) 2022 Hao Luan 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import casadi as ca
import numpy as np
import time
import copy


class NMPC:
    """A Nonlinear Model Predictive Control Class implemented with CasADi. 

    A module for solving a series of (nonlinear) model predictive control problems. 
    By setting up the model, the optimization formulation, and the numerical solver 
    settings, this module can repeatedly solve the (N)MPC problems given any 
    parameters, serving as a closed-loop planner or controller. 

    Example: 
        .. highlight:: python
        .. code-block:: python

            mpc = NMPC() 
            mpc.add_dyn_constraints(dyn=system_dynamics) 
            mpc.add_obstacle_avoidance_constraints(obstacles=obstacle_list) 
            mpc.formulate_cost()
            mpc.set_solver()
            mpc.set_parameters(curr_state, goal_state) 
            mpc.solve() 

    Attributes: 
        default_params (dict): default parameters for this NMPC solver. 
        default_opt_setting (dict): default numerical optimizer settings (IPOPT). 

        NT (int): Look-ahead horizon for MPC. 
        DT (int): Discretization time resolution. 
        NX (int): Dimension of the state. 
        NU (int): Dimension of the input. 
        N_CBF (int): The horizon within which the CBF constraints should be respected. 
        v_max (float or :class:`numpy.array`): Max velocities. 
        a_max (float or :class:`numpy.array`): Max acceleration. 
        P (:class:`numpy.ndarray`): Quadratic terminal state cost matrix 
            (:math:`NX \\times NX`). 
        Q (:class:`numpy.ndarray`): Quadratic stage state cost matrix 
            (:math:`NX \\times NX`). 
        R (:class:`numpy.ndarray`): Quadratic stage input cost matrix 
            (:math:`NU \\times NU`). 
        R_d (:class:`numpy.ndarray`): Quadratic stage input change cost matrix 
            (:math:`NU \\times NU`). 
        gamma_k (float): CBF constraints coefficient. Decaying rate: 
            :math:`1-\gamma_k`.  
        timing (bool): True for timing each MPC solving iteration and print the time. 
        u0 (:class:`numpy.array`): Initial guess for the first step of solution. 
        params_dict (dict): The user input parameter dict. 

        opti (:class:`casadi.Opti`): CasADi's Opti stack. 
        
        opt_x0 (CasADi symbolic parameter): Initial state. 
        opt_xf (CasADi symbolic parameter): Terminal state. 
        opt_P (CasADi symbolic parameter): Quadratic terminal state cost matrix. 
        opt_Q (CasADi symbolic parameter): Quadratic stage state cost matrix.
        opt_R (CasADi symbolic parameter): Quadratic stage input change cost matrix. 
        opt_R_d (CasADi symbolic parameter): Quadratic stage input change cost matrix. 

        opt_states (CasADi symbolic variable): State trajectory decision variable. 
        opt_controls (CasADi symbolic variable): Control trajectory decision variable. 
        opt_obj (CasADi symbolic variable): The cost function. 

        opt_solution (CasADi solution): The solution returned by `opti.solve()`. 

        x_reslt (:class:`numpy.array`): The first step of the numerical state solution. 
        u_reslt (:class:`numpy.array`): The first step of the numerical input solution. 

    """
    default_params={ 
        "NT": 30,                       # horizon [1]
        "NX": 3,                        # state size [1]
        "NU": 3,                        # input size [1]
        "DT": 0.2,                      # discretized time step [s]
        "N_CBF": 3,                     # CBF constraint horizon [1]
        "MAX_VEL": 1.5*np.ones(3),      # maximum velocity [m/s]
        "MAX_ACC": 1.0,                 # maximum acceleration [m/s^2]
        "MAX_DSTEER": np.deg2rad(15.0), # maximum steering speed [rad/s] 
        "P": np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 2.0]
        ]),
        "Q": np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.1]
        ]),
        "R": np.array([
            [0.1, 0.0, 0.0], 
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.03],
        ]),
        "R_d": np.array([
            [0.1, 0.0, 0.0], 
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.03],
        ]),                             # steer change cost
        "gamma_k": .2 ,
        "tictoc": False,
    }

    default_opt_setting={
        'ipopt.max_iter':200, 
        'ipopt.print_level':0, 
        'print_time':0, 
        'ipopt.acceptable_tol':1e-3, 
        'ipopt.acceptable_obj_change_tol':1e-3,
    }


    @staticmethod
    def cbf_circle(st, ob, safe_d=0.80):
        """ A circle-range control barrier function candidate. 

        The value function is defined as 
        
        .. math:: 

            h := (x-x_{\\textrm{ob}})^2 + (y-y_{\\textrm{ob}})^2 - d^2  . 

        Args:
            st: Current agent state. 
            ob: Obstacle position. 
            safe_d (float): safe distance. default is 0.6

        Returns:
            The value of the CBF candidate. 
        """
        return (st[0]-ob[0])**2 + (st[1]-ob[1])**2 - safe_d**2 
    

    def __init__(self, params=default_params):
        """
        """
        ## parameters for optimization
        self.timing = params["tictoc"] if "tictoc" in params else self.default_params["tictoc"]
        self.NT = params["NT"] if "NT" in params else self.default_params["NT"]  # MPC horizon
        self.DT = params["DT"] if "DT" in params else self.default_params["DT"]
        self.NX = params["NX"] if "NX" in params else self.default_params["NX"]
        self.NU = params["NU"] if "NU" in params else self.default_params["NU"] 
        self.N_CBF = params["N_CBF"] if "N_CBF" in params else self.default_params["N_CBF"]
        self.v_max = params["MAX_VEL"] if "MAX_VEL" in params else self.default_params["MAX_VEL"]
        self.a_max = params["MAX_ACC"] if "MAX_ACC" in params else self.default_params["MAX_ACC"]
        self.P = params["P"] if "P" in params else self.default_params["P"]
        self.Q = params["Q"] if "Q" in params else self.default_params["Q"]
        self.R = params["R"] if "R" in params else self.default_params["R"]
        self.R_d = params["R_d"] if "R_d" in params else self.default_params["R_d"] # control change cost 
        self.gamma_k = params["gamma_k"] if "gamma_k" in params else self.default_params["gamma_k"] 
        self.gamma_k = np.clip(self.gamma_k, 1e-4, 1-1e-4)

        self.u0 = np.zeros(self.NU)

        self.params_dict = params

        ## Optization construction
        self.opti = ca.Opti()

        # Parameters
        self.opt_x0 = self.opti.parameter(self.NX)  # (NX,1)
        self.opt_xf = self.opti.parameter(self.NX)  # (NX,1)
        self.opt_P = self.opti.parameter(self.NX, self.NX)
        self.opt_Q = self.opti.parameter(self.NX, self.NX)
        self.opt_R = self.opti.parameter(self.NU, self.NU)
        self.opt_R_d = self.opti.parameter(self.NU, self.NU)

        # Variables
        self.opt_controls = self.opti.variable(self.NT, self.NU)
        self.opt_states = self.opti.variable(self.NT, self.NX)

        # Solution
        self.opt_solution = None
        
        # Results
        self.x_reslt = np.zeros(self.NX)
        self.u_reslt = np.zeros(self.NU)


    def add_dyn_constraints(self, dyn = None):
        """ Adding agent's dynamics constraints for first-order integrators. 

        Args:
            dyn (:obj:`function`): A function (implemented with CasADi) representing
                the agent's dynamics. 
        """
        assert dyn is not None

        vx = self.opt_controls[:, 0]
        vy = self.opt_controls[:, 1]
        omega = self.opt_controls[:, 2]

        # Initial state constraint
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T)

        # Admissable Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max[0], vx, self.v_max[0]))
        self.opti.subject_to(self.opti.bounded(-self.v_max[1], vy, self.v_max[1]))
        self.opti.subject_to(self.opti.bounded(-self.v_max[2], omega, self.v_max[2]))

        # System Model constraints
        for i in range(self.NT-1):
            x_next = dyn(self.opt_states[i,:], self.opt_controls[i,:], params=self.params_dict)
            self.opti.subject_to(self.opt_states[i+1, :]==x_next)
    
    def add_diff_dyn_constraints(self, dyn = None):
        """ Adding agent's dynamics constraints for first-order integrators. 

        Args:
            dyn (:obj:`function`): A function (implemented with CasADi) representing
                the agent's dynamics. 
        """
        assert dyn is not None

        vx = self.opt_controls[:, 0]
        omega = self.opt_controls[:, 1]

        # Initial state constraint
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T)

        # Admissable Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max[0], vx, self.v_max[0]))
        self.opti.subject_to(self.opti.bounded(-self.v_max[1], omega, self.v_max[1]))

        # System Model constraints
        for i in range(self.NT-1):
            x_next = dyn(self.opt_states[i,:], self.opt_controls[i,:], params=self.params_dict)
            self.opti.subject_to(self.opt_states[i+1, :]==x_next) 


    def add_dyn_constraints_2(self, dyn = None):
        """ Adding agent's dynamics constraints for second-order integrators. 

        Args:
            dyn (:obj:`function`): A function (implemented with CasADi) representing
                the agent's dynamics. 
        """
        assert dyn is not None

        vx = self.opt_states[:, 3]
        vy = self.opt_states[:, 4]
        omega = self.opt_states[:, 5]

        ax = self.opt_controls[:, 0]
        ay = self.opt_controls[:, 1]
        alpha = self.opt_controls[:, 2]

        # Initial state constraint
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0[:].T)

        # State constraints
        ## velocities
        self.opti.subject_to(self.opti.bounded(-self.v_max[0], vx, self.v_max[0]))
        self.opti.subject_to(self.opti.bounded(-self.v_max[1], vy, self.v_max[1]))
        self.opti.subject_to(self.opti.bounded(-self.v_max[2], omega, self.v_max[2]))

        # Admissable Control constraints
        ## accelerations
        self.opti.subject_to(self.opti.bounded(-self.a_max[0], ax, self.a_max[0]))
        self.opti.subject_to(self.opti.bounded(-self.a_max[1], ay, self.a_max[1]))
        self.opti.subject_to(self.opti.bounded(-self.a_max[2], alpha, self.a_max[2]))

        # System Model constraints
        for i in range(self.NT-1):
            x_next = dyn(self.opt_states[i,:], self.opt_controls[i,:], params=self.params_dict) 
            self.opti.subject_to(self.opt_states[i+1, :]==x_next) 


    def add_obstacle_avoidance_constraints(self, obstacles:list):
        """ Adding obstacle avoidance constraints.  

        Args:
            obstacles (list): States of the obstacles of interest. 
        """
        if obstacles is None or len(obstacles)<1:
            # print("No obstacles nearby.")
            return
        print("Obstacles nearby count:", len(obstacles)) 

        # CBF circle obstacle avoidance
        h = NMPC.cbf_circle
        for i in range(self.N_CBF):
            for ob in obstacles:
                # print(ob)
                self.opti.subject_to( h(self.opt_states[i+1, :3], ob) >= (1-self.gamma_k) *h(self.opt_states[i, :3], ob) ) 
                # print("h1: {}".format(h(self.opt_states[i+1, :3], ob)))
                # print("h2: {}".format((1-self.gamma_k) *h(self.opt_states[i, :3], ob)))
                # self.opti.subject_to(h(self.opt_states[i, :3], ob)>=0) 

    def formulate_cost(self):
        """ Define the cost function. 
        """
        # Cost function
        ## Stage costs
        self.opt_obj = 0 
        for i in range(self.NT): 
            ### state difference costs
            self.opt_obj = self.opt_obj + ca.mtimes([(self.opt_states[i, :] - self.opt_xf.T), self.opt_Q, (self.opt_states[i, :]- self.opt_xf.T).T])
            ### control costs
            self.opt_obj = self.opt_obj + ca.mtimes([self.opt_controls[i, :], self.opt_R, self.opt_controls[i, :].T]) 
            ### control change penalty
            du = self.opt_controls[i, :] - self.opt_controls[i-1, :] if i>0 else self.opt_controls[i, :] - ca.MX(self.u0[:]).T  
            self.opt_obj = self.opt_obj + ca.mtimes([du, self.opt_R_d, du.T]) 

        ## Terminal costs 
        self.opt_obj = self.opt_obj + ca.mtimes([(self.opt_states[self.NT-1, :] - self.opt_xf.T), self.opt_P, (self.opt_states[self.NT-1, :]- self.opt_xf.T).T]) 

        self.opti.minimize(self.opt_obj)


    def set_solver(self, opts_setting = default_opt_setting):
        """ Setup the numerical solver
        """
        # Solver setting
        self.opti.solver('ipopt', opts_setting)


    def set_parameters(self, state, goal):
        """ Setup parameters 

        Args:
            state (1D array-like): Current state. 
            goal (1D array-like): Goal state. 
        
        Todo: 
            * Provide more parameters setup options. 
        """
        # Set parameter values
        self.opti.set_value(self.opt_x0, np.array([state]))
        self.opti.set_value(self.opt_xf, np.array([goal]))

        self.opti.set_value(self.opt_P, self.P)
        self.opti.set_value(self.opt_Q, self.Q)
        self.opti.set_value(self.opt_R, self.R)
        self.opti.set_value(self.opt_R_d, self.R_d)


    def warmstart(self, state=None, u0=None):
        """ Warmstart the solver. 

        Provide initial guesses for the solver to expedite convergence. 
        
        Args: 
            state (1D array-like): An initial guess for the first step of the 
                state decision variale. 
            u0 (1D array-like): An initial guess for the first step of the 
                input decision variale. 

        Todo: 
            * Passing initials: 
                `opti.set_initial(sol.value_variables())` 

        """
        # Warmstarting
        if u0 is not None:
            self.u0 = u0 
            self.opti.set_initial(self.opt_controls[0, :], self.u0) 
        if state is not None:
            self.opti.set_initial(self.opt_states[0, :], np.array(state)) 


    def solve(self):
        """ Solve the optimization! 

        Returns: 
            True if the solver succeeds; 
            False otherwise. 
        """
        is_success = True
        if self.timing:
            tic = time.time()
        # Solve!
        try:
            self.opt_solution = self.opti.solve()
            self.x_reslt = self.opt_solution.value(self.opt_states) 
            self.u_reslt = self.opt_solution.value(self.opt_controls) 
        except Exception as err:
            is_success = False
            print(type(err))
            print(err.args)
            self.x_reslt = self.opti.debug.value(self.opt_states)
            self.u_reslt = self.opti.debug.value(self.opt_controls)
        
        if self.timing:
            toc = time.time()
            t_used = toc-tic
            print('MPC time spent: %.2f' % (1000*round(t_used, 5)), 'ms')

        return is_success 


    def copy(self):
        """ Deep copy. 

        Returns:
            A deep copy of this NMPC class instance. 
        """
        return copy.deepcopy(self)



########################################################################################
########################################## MAIN ########################################
########################################################################################
def pi2pi(angle):
    z = angle
    while z>=np.pi:
        z -= 2*np.pi
    while z<-np.pi:
        z += np.pi
    return z

def omni_dynamics_ca(st, u, 
    params={
        "DT": .2, 
        # "L": np.sqrt(2) 
    }
):
    DT = params["DT"] if "DT" in params else .2
    # L = params["L"] if "L" in params else np.sqrt(2)
    x_dot = u[0]
    y_dot = u[1]
    theta_dot = u[2]
    st_next = st + DT * ca.vertcat(*[x_dot, y_dot, theta_dot]).T 

    return st_next 

def omni_dynamics(st, u, 
    params={
        "DT": .2, 
        # "L": np.sqrt(2) 
    }
):
    DT = params["DT"] if "DT" in params else .2
    # L = params["L"] if "L" in params else np.sqrt(2)
    x_dot = u[0]
    y_dot = u[1]
    theta_dot = u[2]
    st_next = st + DT * np.array([x_dot, y_dot, theta_dot])

    return st_next 

def omni_dynamics_2ord(st, u, 
    params={
        "DT": .2, 
    }
):
    """ Dynamics of the omnidirectional 2nd-order model (implemented using CasADi). 

    The discrete-time dynamic model is described as 
    
    .. math::

        \\mathbf x_{k+1} = A \\mathbf x_k + B \\mathbf u_k , 
    
    where 

    .. math::

        A = \\begin{bmatrix}
            1& 0& 0& \\Delta t& 0& 0\\\\
            0& 1& 0& 0& \\Delta t& 0\\\\
            0& 0& 1& 0& 0& \\Delta t\\\\
            0& 0& 0& 1& 0& 0\\\\
            0& 0& 0& 0& 1& 0\\\\
            0& 0& 0& 0& 0& 1 
        \\end{bmatrix}, \\quad
        B = \\begin{bmatrix}
            0.5\\Delta t^2&        0&         0\\\\
            0& 0.5\\Delta t^2&         0\\\\
            0&        0&  0.5\\Delta t^2\\\\
            \\Delta t&        0&         0\\\\
            0&       \\Delta t&         0\\\\
            0&        0&        \\Delta t\\\\
        \\end{bmatrix}


    Args: 
        st (CasADi row vector): Current state. 
        u (CasADi row vector): Control input. 
        params : Other parameters. 

    Returns:
        CasADi row vector: Next state. 
    """
    DT = params["DT"] if "DT" in params else .2
    NX = params["NX"] if "NX" in params else 6
    NU = params["NU"] if "NU" in params else 3
    assert NX == NU*2
    assert DT>0 
    A = np.vstack([np.hstack([np.zeros((NX//2, NX//2)), np.eye(NX//2)]), np.zeros((NX//2, NX))]) 
    A = DT * A + np.eye((NX))
    B = DT * np.vstack([0.5*DT*np.eye(NX//2, NU), np.eye(NU)]) 
    # A = np.array([
    #     [1, 0, 0, DT, 0, 0],
    #     [0, 1, 0, 0, DT, 0],
    #     [0, 0, 1, 0, 0, DT],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1],
    # ])
    # B = np.array([
    #     [0.5*DT^2,        0,         0],
    #     [       0, 0.5*DT^2,         0],
    #     [       0,        0,  0.5*DT^2],
    #     [      DT,        0,         0],
    #     [       0,       DT,         0],
    #     [       0,        0,        DT],
    # ])

    st_next = ca.mtimes([A, st.T]) + ca.mtimes([B, u.T]) 


def main():
    import time
    import matplotlib.pyplot as plt

    st = np.array([0,0,0])
    gl = np.array([9, 8.8*np.sqrt(2), np.pi*2/3])

    params={
        "NT": 20,                        # horizon [1]
        "NX": 3,
        "NU": 3,
        "DT": 0.1,                      # discretized time step [s]
        "MAX_VEL": 1.5*np.ones(3),                 # maximum velocity [m/s]
        "MAX_ACC": 1.0,                 # maximum acceleration [m/s^2]
        "MAX_STEER": np.deg2rad(45.0),  # maximum steering [rad]
        "MAX_DSTEER": np.deg2rad(30.0), # maximum steering speed [rad/s] 
        "L": np.sqrt(2),                # car length [s]
        "Q": np.array([
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.5], 
                ]),
        "R": np.array([
                        [0.01, 0.0, 0], 
                        [0.0, 0.01, 0],
                        [0.0, 0.0, 0.001],
                    ]),
        "R_d": np.array([
                        [0.01, 0.0, 0], 
                        [0.0, 0.01, 0],
                        [0.0, 0.0, 0.001],
                    ]),                   # steer change cost
        "tictoc": False,
    }

    opts_setting = {
        'ipopt.max_iter':200, 
        'ipopt.print_level':0, 
        'print_time':0, 
        'ipopt.acceptable_tol':1e-4, 
        'ipopt.acceptable_obj_change_tol':1e-3,
    }

    isPrint = True

    t = 0
    T = 10
    dt = params["DT"]
    ctrl = np.zeros(3)
    trj = st
    cmds = ctrl

    mpc = NMPC(params=params)
    mpc.add_dyn_constraints(dyn = omni_dynamics_ca)
    mpc.formulate_cost()
    mpc.set_solver(opts_setting=opts_setting)

    while (np.linalg.norm(st[:2]-gl[:2])>.01 or abs(np.rad2deg(st[2]-gl[2]))>5) and t<T:
        tic = time.time()

        mpc.set_parameters(st, gl)
        # mpc.warmstart(st, ctrl)
        mpc.solve()
        toc = time.time()
        t_used = abs(toc - tic)
        if isPrint:
            print('t_used = %.2f' % (1000*round(t_used, 5)), 'ms')

        ctrl = mpc.u_reslt[0]
        st_mat = np.array(omni_dynamics(st, ctrl, params))
        st = st_mat.flatten()
        st[2] = pi2pi(st[2])

        t += dt 
        if isPrint:
            print(mpc.x_reslt)
            print("x[%.2f] = " % (t), st)
            print("u[%.2f] = " % (t), ctrl, '\n')

        trj = np.vstack((trj, st))
        cmds = np.vstack((cmds, ctrl))

    # Plot
    t_stamp=np.linspace(0,t,trj.shape[0])

    fig1, axs1 = plt.subplots(6, 1)
    fig1.suptitle("Car")
    for idx in range(3):
        axs1[idx].plot(t_stamp, trj[:, idx]) 
        axs1[idx].grid()
    axs1[0].set_ylabel("$x$")
    axs1[1].set_ylabel("$y$")
    axs1[2].set_ylabel("$\\theta$")
    for idx in range(3):
        axs1[3+idx].plot(t_stamp, cmds[:, idx]) 
    axs1[3].set_ylabel("$v_x$")
    axs1[4].set_ylabel("$v_y$")
    axs1[5].set_ylabel("$\\omega$")

    fig, ax = plt.subplots() 
    ax.plot(trj[:,0], trj[:,1], label="Car")
    ax.scatter(trj[0,0], trj[0,1], marker="D")
    ax.scatter(gl[0], gl[1], marker="*")

    ax.grid()
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("$X$-$Y$ Relation")

    plt.show()


if __name__ == '__main__':
    main()
