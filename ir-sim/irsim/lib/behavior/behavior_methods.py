from irsim.lib import register_behavior
from irsim.lib import reciprocal_vel_obs
from irsim.lib import optimal_reciprocal_col_avoid
from irsim.lib import social_force_model
from irsim.util.util import relative_position, WrapToPi, omni_to_diff
import numpy as np
from math import cos, sin

@register_behavior("diff", "dash")
def beh_diff_dash(ego_object, external_objects, **kwargs):

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)

    behavior_vel = DiffDash(state, goal, max_vel, goal_threshold, angle_tolerance)

    return behavior_vel

@register_behavior("omni", "dash")
def beh_omni_dash(ego_object, external_objects, **kwargs):

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    behavior_vel = OmniDash(state, goal, max_vel, goal_threshold)

    return behavior_vel

@register_behavior("diff", "rvo")
def beh_diff_rvo(ego_object, external_objects, **kwargs):
    is_target = (ego_object.role == "target")  # supposing target is not influenced by robot
    rvo_neighbor = []
    for obj in external_objects:
        if is_target and obj.role == "robot":
            continue
        rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    vxmax = kwargs.get("vxmax", 1.5)
    vymax = kwargs.get("vymax", 1.5)
    acceler = kwargs.get("acceler", 1.0)
    factor = kwargs.get("factor", 1.0)
    mode = kwargs.get("mode", "rvo")
    behavior_vel = DiffRVO(rvo_state, rvo_neighbor, vxmax, vymax, acceler, factor, mode)

    return behavior_vel

@register_behavior("omni", "rvo")
def beh_omni_rvo(ego_object, external_objects, **kwargs):
    is_target = (ego_object.role == "target")  # supposing target is not influenced by robot
    rvo_neighbor = []
    for obj in external_objects:
        if is_target and obj.role == "robot":
            continue
        rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    vxmax = kwargs.get("vxmax", 1.5)
    vymax = kwargs.get("vymax", 1.5)
    acceler = kwargs.get("acceler", 1.0)
    factor = kwargs.get("factor", 1.0)
    mode = kwargs.get("mode", "rvo")
    behavior_vel = OmniRVO(rvo_state, rvo_neighbor, vxmax, vymax, acceler, factor, mode)

    return behavior_vel

@register_behavior("diff", "orca")
def beh_diff_orca(ego_object, external_objects, **kwargs):
    is_target = (ego_object.role == "target")  # supposing target is not influenced by robot
    rvo_neighbor = []
    for obj in external_objects:
        if is_target and obj.role == "robot":
            continue
        rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    behavior_vel = DiffORCA(rvo_state, rvo_neighbor, **kwargs)

    return behavior_vel

@register_behavior("omni", "orca")
def beh_omni_orca(ego_object, external_objects, **kwargs):
    is_target = (ego_object.role == "target")  # supposing target is not influenced by robot
    rvo_neighbor = []
    for obj in external_objects:
        if is_target and obj.role == "robot":
            continue
        rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    behavior_vel = OmniORCA(rvo_state, rvo_neighbor, **kwargs)

    return behavior_vel

@register_behavior("diff", "sfm")
def beh_diff_sfm(ego_object, external_objects, **kwargs):
    is_target = (ego_object.role == "target")  # supposing target is not influenced by robot
    rvo_neighbor = []
    for obj in external_objects:
        if is_target and obj.role == "robot":
            continue
        rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    goal = ego_object.goal
    behavior_vel = DiffSFM(rvo_state, rvo_neighbor, goal, **kwargs)

    return behavior_vel

@register_behavior("omni", "sfm")
def beh_omni_sfm(ego_object, external_objects, **kwargs):
    is_target = (ego_object.role == "target")  # supposing target is not influenced by robot
    rvo_neighbor = []
    for obj in external_objects:
        if is_target and obj.role == "robot":
            continue
        rvo_neighbor.append(obj.rvo_neighbor_state)
    rvo_state = ego_object.rvo_state
    goal = ego_object.goal
    behavior_vel = OmniSFM(rvo_state, rvo_neighbor, goal, **kwargs)

    return behavior_vel

@register_behavior("acker", "dash")
def beh_acker_dash(ego_object, external_objects, **kwargs):

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)

    behavior_vel = AckerDash(state, goal, max_vel, goal_threshold, angle_tolerance)

    return behavior_vel

def DiffSFM(
    state_tuple,
    neighbor_list=None,
    goal=None,
    **kwargs
):
    """
    Calculate the diff velocity using SFM.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acceler (float): Acceleration factor (default 1).

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """
    if neighbor_list is None:
        neighbor_list = []
    # print("goal:", goal)
    sfm_behavior = social_force_model(
        state_tuple, neighbor_list, goal, **kwargs
    )
    sfm_vel = sfm_behavior.cal_vel()
    sfm_vel_diff = omni_to_diff(state_tuple[-1], sfm_vel)

    return sfm_vel_diff

def OmniSFM(
    state_tuple,
    neighbor_list=None,
    goal=None,
    **kwargs
):
    """
    Calculate the omnidirectional velocity using RVO.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acceler (float): Acceleration factor (default 1).

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """
    if neighbor_list is None:
        neighbor_list = []
    # print("goal:", goal)
    sfm_behavior = social_force_model(
        state_tuple, neighbor_list, goal, **kwargs
    )
    sfm_vel = sfm_behavior.cal_vel()

    return np.array([[sfm_vel[0]], [sfm_vel[1]]])

def DiffORCA(
    state_tuple,
    neighbor_list=None,
    **kwargs,
):
    """
    Calculate the omnidirectional velocity using RVO.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acceler (float): Acceleration factor (default 1).

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """
    if neighbor_list is None:
        neighbor_list = []

    orca_behavior = optimal_reciprocal_col_avoid(
        state_tuple, neighbor_list, **kwargs
    )
    orca_vel = orca_behavior.cal_vel()
    orca_vel_diff = omni_to_diff(state_tuple[-1], orca_vel)

    return orca_vel_diff

def OmniORCA(
    state_tuple,
    neighbor_list=None,
    **kwargs,
):
    """
    Calculate the omnidirectional velocity using RVO.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acceler (float): Acceleration factor (default 1).

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """
    if neighbor_list is None:
        neighbor_list = []

    orca_behavior = optimal_reciprocal_col_avoid(
        state_tuple, neighbor_list, **kwargs
    )
    orca_vel = orca_behavior.cal_vel()

    return np.array([[orca_vel[0]], [orca_vel[1]]])

def DiffRVO(
    state_tuple,
    neighbor_list=None,
    vxmax=1.5,
    vymax=1.5,
    acceler=1,
    factor=1.0,
    mode="rvo",
):
    """
    Calculate the differential drive velocity using RVO.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acceler (float): Acceleration factor (default 1).
        factor (float): Additional scaling factor (default 1.0).
        mode (str): RVO calculation mode (default "rvo").

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    if neighbor_list is None:
        neighbor_list = []

    rvo_behavior = reciprocal_vel_obs(
        state_tuple, neighbor_list, vxmax, vymax, acceler, factor
    )
    rvo_vel = rvo_behavior.cal_vel(mode)
    rvo_vel_diff = omni_to_diff(state_tuple[-1], rvo_vel)

    return rvo_vel_diff

def OmniRVO(
    state_tuple,
    neighbor_list=None,
    vxmax=1.5,
    vymax=1.5,
    acceler=1,
    factor=1.0,
    mode="rvo",
):
    """
    Calculate the omnidirectional velocity using RVO.

    Args:
        state_tuple (tuple): Current state and orientation.
        neighbor_list (list): List of neighboring agents (default None).
        vxmax (float): Maximum x velocity (default 1.5).
        vymax (float): Maximum y velocity (default 1.5).
        acceler (float): Acceleration factor (default 1).
        factor (float): Additional scaling factor (default 1.0).
        mode (str): RVO calculation mode (default "rvo").

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """
    if neighbor_list is None:
        neighbor_list = []

    rvo_behavior = reciprocal_vel_obs(
        state_tuple, neighbor_list, vxmax, vymax, acceler, factor
    )
    rvo_vel = rvo_behavior.cal_vel(mode)

    return np.array([[rvo_vel[0]], [rvo_vel[1]]])

def DiffDash(state, goal, max_vel, goal_threshold=0.1, angle_tolerance=0.2):
    """
    Calculate the differential drive velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y, theta] (3x1).
        max_vel (np.array): Maximum velocity [linear, angular] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).
        angle_tolerance (float): Allowable angular deviation (default 0.2).

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    distance, radian = relative_position(state, goal)

    if distance < goal_threshold:
        return np.zeros((2, 1))

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = max_vel[0, 0] * np.cos(diff_radian)

    if abs(diff_radian) < angle_tolerance:
        angular = 0
    else:
        angular = max_vel[1, 0] * np.sign(diff_radian)

    return np.array([[linear], [angular]])

def OmniDash(state, goal, max_vel, goal_threshold=0.1):
    """
    Calculate the omnidirectional velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y] (2x1).
        goal (np.array): Goal position [x, y] (2x1).
        max_vel (np.array): Maximum velocity [vx, vy] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).

    Returns:
        np.array: Velocity [vx, vy] (2x1).
    """
    distance, radian = relative_position(state, goal)

    if distance > goal_threshold:
        vx = max_vel[0, 0] * cos(radian)
        vy = max_vel[1, 0] * sin(radian)
    else:
        vx = 0
        vy = 0

    return np.array([[vx], [vy]])

def AckerDash(state, goal, max_vel, goal_threshold, angle_tolerance):
    """
    Calculate the Ackermann steering velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y, theta] (3x1).
        max_vel (np.array): Maximum velocity [linear, steering angle] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached.
        angle_tolerance (float): Allowable angular deviation.

    Returns:
        np.array: Velocity [linear, steering angle] (2x1).
    """
    dis, radian = relative_position(state, goal)
    steer_opt = 0.0
    diff_radian = WrapToPi(radian - state[2, 0])

    if diff_radian > -angle_tolerance and diff_radian < angle_tolerance:
        diff_radian = 0

    if dis < goal_threshold:
        v_opt, steer_opt = 0, 0
    else:
        v_opt = max_vel[0, 0]
        steer_opt = np.clip(diff_radian, -max_vel[1, 0], max_vel[1, 0])

    return np.array([[v_opt], [steer_opt]])
