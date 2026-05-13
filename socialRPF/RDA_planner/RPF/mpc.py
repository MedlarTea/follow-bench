import numpy as np
import yaml
from .omni_nmpc_casadi_oop import NMPC as NMPC_O
from .diff_nmpc_casadi_oop import NMPC as NMPC_D
from .omni_nmpc_casadi_oop import omni_dynamics_ca, omni_dynamics_2ord
from .diff_nmpc_casadi_oop import diff_dynamics_ca
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
class MPC:
    def __init__(self, dyn_type='omni', ctrl_type=1):
        """
        Input:
            dyna_type: omni or diff
            ctrl: 1 or 2, representing velocity or acceleration control
        """
        
        with open(os.path.join(current_dir, '{}.yaml'.format(dyn_type)), 'r') as file:
            config = yaml.safe_load(file)


        self.ctrl_type = ctrl_type
        self.dyn_type = dyn_type
        if self.ctrl_type not in [1, 2]:
            raise RuntimeError("Unknown control type!")
        if self.dyn_type not in ['omni', 'diff']:
            raise RuntimeError("Unknown dyanmics type!")

        self.plan_config = {
            "NT": config["NT"],                   # horizon [1]
            "NX": config["NX"],
            "NU": config["NU"],
            "N_CBF": config["N_CBF"],
            "DT": config["DT"],                   # discretized time step [s] 
            "MAX_VEL": np.array([
                config["max_vx"],                # maximum x velocity [m/s]
                config["max_vy"],                # maximum y velocity [m/s]
                config["max_va"],                # maximum angle velocity [rad/s]
            ]),
            "MAX_ACC": np.array([
                config["max_ax"],                # maximum x acceleration [m/s^2]
                config["max_ay"],                # maximum y acceleration [m/s^2]
                config["max_aa"],                # maximum angle acceleration [rad/s^2]
            ]),
            "P": np.diag(config["P"]),
            "Q": np.diag(config["Q"]),
            "R": np.diag(config["R"]),
            "R_d": np.diag(config["R_d"]),        # steer change cost
            "gamma_k": config["gamma_k"],
            "tictoc": config["tictoc"],
            "omega": config["omega"],
            "sigma": config["sigma"],
            "safe_distance": config["safe_distance"]
}

        self.plan_solver_config = config["plan_solver_config"]

        self.planned_trj = np.zeros((20, self.plan_config["NX"]))
        self.ctrl_inputs = np.zeros(self.plan_config['NU']) 

        self.robot_pose = np.zeros(3)
        self.goal_pose = np.zeros(3)


        # Add kinematic constraints
        if self.dyn_type == 'diff':
            self.mpc_plate = NMPC_D(params=self.plan_config)
            if self.ctrl_type == 1:
                self.mpc_plate.add_dyn_constraints(dyn=diff_dynamics_ca)
            else:
                raise ModuleNotFoundError('just implemented 1 order control yet')
        elif self.dyn_type == 'omni':
            self.mpc_plate = NMPC_O(params=self.plan_config)
            if self.ctrl_type == 1:
                self.mpc_plate.add_dyn_constraints(dyn=omni_dynamics_ca)
            if self.ctrl_type == 2:
                self.mpc_plate.add_dyn_constraints(dyn=omni_dynamics_2ord)
        else:
            raise RuntimeError("Unkown dynamics type!")
        
        ### Add cost function ###
        self.mpc_plate.formulate_cost()  
        # Set up the solver
        self.mpc_plate.set_solver(self.plan_solver_config)

    def mpc_planning(self, mpc):
        """ The planning process. 

        Set up the planner and solve the optimization. 

        Args: 
            mpc (:class:`nmpc_casadi_oop.NMPC`): An MPC planner instance 
                for planning. 

        Raises:
            RuntimeWarning: Unknown dynamics type. 
        """
        if abs(self.goal_pose[0]) < 0.05:
            self.vel = np.zeros(self.plan_config['NU'])
            return 

        if self.ctrl_type == 1:
            mpc.set_parameters(self.robot_pose, self.goal_pose)
            # mpc.warmstart(self.pose, self.ctrl_inputs[0])
        elif self.ctrl_type == 2:
            mpc.set_parameters(np.concatenate((self.robot_pose, self.vel)), self.goal_pose) 
            # mpc.warmstart(np.concatenate((self.pose, self.vel)), self.ctrl_inputs[0]) 
        else:
            raise RuntimeWarning('Unknown dynamics type! Stop planning... ')
        #end ifelse
        flg_success = mpc.solve()

        if not flg_success:
            print("Solver failed! Solutions' NOT reliable!") 
            self.vel = np.zeros(self.plan_config['NU'])
            return 

        self.planned_trj = mpc.x_reslt
        # print('x=',mpc.x_reslt)
        self.ctrl_inputs = mpc.u_reslt
        # print('u=',mpc.u_reslt)
        if self.ctrl_type == 1:
            self.vel = self.ctrl_inputs[0]
        elif self.ctrl_type == 2:
            self.vel = self.planned_trj[1, 3:]  
        else:
            raise RuntimeWarning('Unknown dynamics type! Stoping the vehicle... ')
        return self.planned_trj

    def control(self, robot_pose, goal_pose, obstacles=[]):
        """ self spin main function. 

        Check whether the target is acquired. Plan and publish velocity 
        commands if the target is acquired; stay put if not. 

        Args:
            planner (:class:`nmpc_casadi_oop.NMPC`): An MPC planner instance 
                for planning. 
        
        Raises:
            RuntimeWarning: Unknown dynamics type. 
        """
        robot_pose = np.array(robot_pose).squeeze()
        goal_pose = np.array(goal_pose).squeeze()
        self.robot_pose = robot_pose
        self.goal_pose =  goal_pose

        # print("robot_pose:", robot_pose)
        # print("goal_pose:", goal_pose)

        mpc_instance = self.mpc_plate.copy()
        if len(obstacles) != 0:
            mpc_instance.add_obstacle_avoidance_constraints(obstacles)

        twist = []
        predicted_traj = self.mpc_planning(mpc_instance)

        # Clip the velocity again for safety. 
        if self.dyn_type == 'omni' and self.ctrl_type == 1:
            twist.append(float(np.clip(self.vel[0], -self.plan_config['MAX_VEL'][0], self.plan_config['MAX_VEL'][0])))
            twist.append(float(np.clip(self.vel[1], -self.plan_config['MAX_VEL'][1], self.plan_config['MAX_VEL'][1])))
            twist.append(float(np.clip(self.vel[2], -self.plan_config['MAX_VEL'][2], self.plan_config['MAX_VEL'][2])))
        elif self.dyn_type == 'diff' and self.ctrl_type == 1:
            twist.append(float(np.clip(self.vel[0], -self.plan_config['MAX_VEL'][0], self.plan_config['MAX_VEL'][0])))
            twist.append(float(np.clip(self.vel[1], -self.plan_config['MAX_VEL'][2], self.plan_config['MAX_VEL'][2])))
        else:
            raise RuntimeWarning('Unknown dynamics type! Stop publishing twist commands... ')
        #end if else
        twist = np.array(twist)[:,np.newaxis]
        output_traj = []
        for i in range(self.plan_config['NT']):
            output_traj.append(self.planned_trj[i, :][:, np.newaxis])
        return twist, output_traj