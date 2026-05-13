import os

import numpy as np
# from replayBuffer import state_to_obs , normalize


class MCTSNode(object):
    def __init__(self, state, params, parent=None, agent = None):
        self.number_of_visits = 1.
        self.value = 0.
        self.depth = 0
        self.state = state
        self.parent = parent
        self.children = []
        self.params = params
        self.action = ''
        self.tree_id = 0
        self.number_of_reaction = 0


    @property
    def n(self):
        return self.number_of_visits 


    def expand(self):
        a = len(self.children)
        if self.state.next_to_move:
            action = self.params["human_acts"][a]
        else:
            action = self.params["robot_acts"][a]
            
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, self.params, parent=self)
        child_node.action = action

        if not self.state.next_to_move and not self.is_safe_to_pass(next_state):
            child_node = None

        self.children.append(child_node)
        return child_node

    def is_safe_to_pass(self, next_state):

        if not self.close_to_human(next_state):
            if not self.any_obs(self.state.state[0,:2], next_state.state[0,:2]):
                return True

        return False

    def close_to_human(self, next_state):
        r = self.params['safety_params']['r']   # radious of the circle around human
        a = self.params['safety_params']['a']   # displacement from center of the circle
    
        alpha = np.arctan2(next_state.state[0,1]-next_state.state[1,1]  ,next_state.state[0,0]-next_state.state[1,0]  ) #  yr-yh , xr-xh
        alpha = np.absolute (alpha - next_state.state[1,2])

        roots = np.roots([1, -2*a*np.cos(alpha), a*a-r*r])
        d_circle = np.max(roots)
        d_actual = np.linalg.norm(next_state.state[0,:2] - next_state.state[1,:2])

        if d_actual > d_circle:
            return False
        else:
            return True

    def any_obs(self, s, sp):
        if self.params['sim']:
            return False
        
        debug_obstacle = os.getenv("FOLLOW_AHEAD_DEBUG_OBSTACLE") == "1"
        dx = (sp[0]-s[0]) * 7
        dy = (sp[1]-s[1]) * 7
        probe_x = s[0] + dx
        probe_y = s[1] + dy

        map_width = self.params['map_width']
        map_height = len(self.params['map_data']) // map_width
        sample_count = 5

        for sample_idx, ratio in enumerate(np.linspace(1.0 / sample_count, 1.0, sample_count), start=1):
            new_x = s[0] + (probe_x - s[0]) * ratio
            new_y = s[1] + (probe_y - s[1]) * ratio
            x = int(np.rint((new_x - self.params['map_origin_x']) / self.params['map_res']))
            y = int(np.rint((new_y - self.params['map_origin_y']) / self.params['map_res']))
            in_bounds = 0 <= x < map_width and 0 <= y < map_height

            if in_bounds:
                cost = self.params['map_data'][int(x + map_width * y)]
            else:
                cost = None

            if debug_obstacle:
                print(
                    "[OBS-PROBE] "
                    f"sample={sample_idx}/{sample_count} "
                    f"s={np.round(s, 3).tolist()} "
                    f"sp={np.round(sp, 3).tolist()} "
                    f"probe=({new_x:.3f}, {new_y:.3f}) "
                    f"grid=({x}, {y}) "
                    f"in_bounds={in_bounds} "
                    f"cost={cost}"
                )

            if cost is not None and cost > 30:
                if debug_obstacle:
                    print(
                        "[OBS] "
                        f"sample={sample_idx}/{sample_count} "
                        f"s={np.round(s, 3).tolist()} "
                        f"sp={np.round(sp, 3).tolist()} "
                        f"probe=({new_x:.3f}, {new_y:.3f}) "
                        f"grid=({x}, {y}) "
                        f"cost={cost}"
                    )
                return True

        return False




    def backpropagate(self):
        value = self.value
        parent= self.parent
        pow = 1
              
        while parent:
            parent.number_of_visits +=1
            parent.value += value * self.params['gamma'] ** pow
            parent = parent.parent
            pow +=1


    def is_fully_expanded(self):
        if self.state.next_to_move == 1:
            return len(self.children) == len(self.params["human_acts"])
        else:
            return len(self.children) == len(self.params["robot_acts"])
        
