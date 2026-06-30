import math
import random
import numpy as np
from typing import List, Any, Optional, Dict
from prism.planner.base_planner import BasePlanner
from prism.utils.trajectory import Trajectory
from prism.tasks.base_task import BaseTask
from prism.world_model.base_world_model import BaseWorldModel
from prism.reward.base_reward import BaseReward
from prism.utils.config import MCTSConfig

class MCTSNode:
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Optional[Any] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Any, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.is_done = False

    def ucb1(self, exploration_constant: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTSPlanner(BasePlanner):
    def __init__(self, config: MCTSConfig):
        self.config = config

    def _get_policy_action(self, state: Any, policy: Any, world_model: BaseWorldModel) -> Any:
        """Helper to get an action from the policy for a given environment state."""
        if policy is None:
            return None
        
        # Extract raw observation from state
        raw_obs = state.gen_obs()
        # Use the environment's own processing logic (e.g., FlatObsWrapper)
        # We need to access the env through the world model
        if hasattr(world_model, 'env'):
            obs_flat = world_model.env.process_obs(raw_obs)
        else:
            # Fallback (should not happen in our current setup)
            return random.choice(range(7))
            
        return policy.act(obs_flat)

    def plan(self, initial_state: Any, task: BaseTask, world_model: BaseWorldModel, reward_fn: BaseReward, policy: Optional[Any] = None) -> List[Trajectory]:
        root = MCTSNode(initial_state)
        
        for _ in range(self.config.n_simulations):
            node = root
            
            # 1. Selection & Expansion
            while not node.is_done:
                possible_actions = list(range(7)) # MiniGrid action space
                untried_actions = [a for a in possible_actions if a not in node.children]
                
                if untried_actions:
                    # Epsilon-greedy choice for expansion
                    if policy is not None and random.random() > self.config.epsilon:
                        p_act = self._get_policy_action(node.state, policy, world_model)
                        if p_act in untried_actions:
                            action = p_act
                        else:
                            action = random.choice(untried_actions)
                    else:
                        action = random.choice(untried_actions)
                    
                    # Expansion step
                    next_state, _, done = world_model.step(node.state, action)
                    new_node = MCTSNode(next_state, parent=node, action=action)
                    new_node.is_done = done
                    node.children[action] = new_node
                    node = new_node
                    break # Expanded a new node, move to rollout
                else:
                    # All actions tried, use UCB1 to select child
                    action = max(node.children.keys(), key=lambda a: node.children[a].ucb1(self.config.exploration_constant))
                    node = node.children[action]

            # 2. Simulation (Rollout) using epsilon-greedy policy guidance
            curr_state = node.state
            curr_done = node.is_done
            depth = 0
            
            rollout_states = [curr_state]
            rollout_actions = []
            
            while not curr_done and depth < self.config.max_depth:
                if policy is not None and random.random() > self.config.epsilon:
                    act = self._get_policy_action(curr_state, policy, world_model)
                else:
                    act = random.choice(range(7))
                
                curr_state, _, curr_done = world_model.step(curr_state, act)
                rollout_states.append(curr_state)
                rollout_actions.append(act)
                depth += 1
            
            # 3. Backpropagation using Trajectory-level reward (LLM)
            full_states = []
            full_actions = []
            
            temp_node = node
            while temp_node:
                full_states.insert(0, temp_node.state)
                if temp_node.action is not None:
                    full_actions.insert(0, temp_node.action)
                temp_node = temp_node.parent
            
            full_states.extend(rollout_states[1:])
            full_actions.extend(rollout_actions)
            
            # Helper to convert state to observation
            def get_obs(state):
                raw_obs = state.gen_obs()
                if hasattr(world_model, 'env'):
                    return world_model.env.process_obs(raw_obs)
                return raw_obs

            full_obs = [get_obs(s) for s in full_states]
            full_traj = Trajectory(states=full_states, actions=full_actions, observations=full_obs)
            total_reward = reward_fn.compute(full_traj, task)
            
            while node:
                node.visits += 1
                node.value += total_reward
                node = node.parent

        return self._extract_top_k(root, task, reward_fn, world_model)

    def _extract_top_k(self, root: MCTSNode, task: BaseTask, reward_fn: BaseReward, world_model: BaseWorldModel) -> List[Trajectory]:
        trajectories = []
        
        def get_obs(state):
            raw_obs = state.gen_obs()
            if hasattr(world_model, 'env'):
                return world_model.env.process_obs(raw_obs)
            return raw_obs
        
        def find_paths(node, current_states, current_actions):
            if not node.children:
                current_obs = [get_obs(s) for s in current_states]
                traj = Trajectory(states=list(current_states), actions=list(current_actions), observations=current_obs)
                traj.total_reward = reward_fn.compute(traj, task)
                trajectories.append(traj)
                return

            sorted_actions = sorted(node.children.keys(), key=lambda a: node.children[a].visits, reverse=True)
            for action in sorted_actions[:3]:
                child = node.children[action]
                find_paths(child, current_states + [child.state], current_actions + [action])

        find_paths(root, [root.state], [])
        return sorted(trajectories, key=lambda x: x.total_reward, reverse=True)[:self.config.n_rollouts_returned]
