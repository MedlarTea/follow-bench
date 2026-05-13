from pathlib import Path

from .nodes import MCTSNode
import numpy as np
import torch

from .follow_task_utils import build_rl_observation
from treelib import Tree
import os
import time


class MCTS:
    def __init__(self, node: MCTSNode, human_prob):
        self.root = node
        self.params = node.params
        self.next_node_candidates = []
        self.human_prob = human_prob


    def tree_expantion(self, T):
        tree_id = 0
        expansion_count = 0

        while time.time() < T:
            expansion_count += 1

            ### Node selection 
            curr_node = self.root
            while(curr_node.number_of_visits > 1):
                #### UCB of the children and selecting a child node
                UCB_children = []
                for c in curr_node.children:
                    UCB_children.append(self.compute_UCB(c))

                curr_node = curr_node.children[np.argmax(UCB_children)]  

            ### at this point current node is a leaf node
            ### Node expantion
            while not curr_node.is_fully_expanded():
                child_node = curr_node.expand()
                if child_node == None:
                    continue
                child_node.value = self.evaluate_node(child_node)
                child_node.backpropagate()
                child_node.tree_id = tree_id+1
                tree_id +=1
            
            # self.draw_tree()
        print('expansion_count:', expansion_count)
        return self.best_child_node()


    def compute_UCB(self, c):
        if not c:
            return -np.inf
        
        c_param= 2.
        prob = 1.
        if c.state.next_to_move == 0: 
            if c.action == 'left':
                prob = self.human_prob['left']
            elif c.action == 'straight':
                prob = self.human_prob['straight']
            elif c.action == 'right':
                prob = self.human_prob['right']

        else:
            prob = 1.      

        UCB = ((c.value/c.n) + c_param * np.sqrt((np.log(c.parent.n) / c.n)) ) * prob

        return UCB


    def draw_tree(self):
        self.tree = Tree()
        node = self.root
        self.tree.create_node("parent_("+str(self.root.value/self.root.number_of_visits)+")", 0)
        self.extent_draw_tree(node)

        path = Path("/tmp/follow_ahead_reaction_tree.txt")
        if os.path.exists(path):
            os.remove(path)

        self.tree.save2file(str(path))

    def extent_draw_tree(self, parent):
        for node in parent.children:
            if node:
                val = node.val if not node.number_of_visits else node.value/node.number_of_visits
                self.tree.create_node(node.action+'_('+str(val)+')',  node.tree_id , node.parent.tree_id)
                self.extent_draw_tree(node)

        return
    
    def evaluate_node(self, node):
        # immidiate reward
        r = node.state.calculate_reward(node.state.state)

        # estimated return
        state = node.state.state
        rel_vec = state[0, :2] - state[1, :2]
        obs = build_rl_observation(
            rel_vec,
            state[1, 2],
            state[0, 2],
            self.params.get('follow_mode', 'front'),
            self.params.get('desired_distance', 1.5),
            obs_mode=self.params.get('rl_obs_mode', 'relative_pose'),
        )
        obs = torch.FloatTensor(obs).unsqueeze(0)
        policy='a2c'
        value = self.params['RL_model'].evaluate_state(obs, policy=policy)

        return  r + value.item() /10 * self.params['gamma']
    
    def best_child_node(self):
        visit = []
        for node in self.root.children:
            if node:
                visit.append(node.n)
            else:
                visit.append(0)
      
        idx = np.argmax(visit)

        if visit[-1] == visit[idx]:
            return self.root.children[-1]
        elif visit[2] == visit[idx]:
            return self.root.children[2]
        elif visit[0] == visit[1] == visit[idx]:
            return self.root.children[2] ## if fast left == fast right then take fast straight
        elif visit[3] == visit[4] == visit[idx]:
            return self.root.children[5]  ## if left == right then take straight
        return self.root.children[np.argmax(visit)]        
