import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from prism.imitation.base_imitation import BaseImitationLearner
from prism.agent.base_agent import BaseAgent
from prism.agent.meta_agent import MetaAgent
from prism.utils.trajectory import Trajectory
from prism.utils.config import BCConfig

class BehavioralCloning(BaseImitationLearner):
    def __init__(self, config: BCConfig):
        self.config = config

    def update(self, agent: BaseAgent, trajectories: List[Trajectory]) -> float:
        if not isinstance(agent, MetaAgent):
            raise ValueError("BehavioralCloning expects a MetaAgent")
        
        policy = agent.policy
        optimizer = optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        policy.train()
        total_loss = 0.0
        
        # Collect all state-action pairs from trajectories
        all_states = []
        all_actions = []
        for traj in trajectories:
            states_to_use = traj.observations if (hasattr(traj, 'observations') and traj.observations) else traj.states
            all_states.extend(states_to_use[:-1]) # states has one more element than actions
            all_actions.extend(traj.actions)
        
        if not all_states:
            return 0.0
            
        import numpy as np
        states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32, device=agent.device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=agent.device)
        
        for _ in range(self.config.n_gradient_steps):
            optimizer.zero_grad()
            logits, _ = policy(states_tensor)
            loss = criterion(logits, actions_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / self.config.n_gradient_steps
