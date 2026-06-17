import torch
import torch.nn as nn
from prism.reward.base_reward import BaseReward
from prism.utils.trajectory import Trajectory
from prism.tasks.base_task import BaseTask
from typing import Any, Optional

class ValueFunctionReward(BaseReward):
    """
    Reward module that uses a pretrained value function for both 
    per-step guidance and trajectory ranking.
    """
    def __init__(self, value_model: nn.Module, device: str = 'cpu'):
        self.value_model = value_model
        self.value_model.eval()
        self.device = device

    def compute(self, trajectory: Trajectory, task: BaseTask) -> float:
        """Trajectory-level reward: max value achieved during the trajectory."""
        if not trajectory.states:
            return 0.0
        
        # In a real implementation, we'd batch the states and pass to value_model
        # For now, we'll assume states are already in a format the model likes
        values = []
        with torch.no_grad():
            for state in trajectory.states:
                # This assumes 'state' is the observation, not the env object
                # We might need to adjust based on what MCTS stores in nodes
                val = self.compute_dense(state, task)
                values.append(val)
        return max(values)

    def compute_dense(self, state: Any, task: BaseTask) -> float:
        """Dense reward: value function estimate for (state, goal)."""
        # This implementation depends on the specific value_model interface
        # For MiniGrid/BabyAI research codebases, this often involves 
        # combining the observation with a goal encoding from the task spec.
        spec = task.get_spec()
        goal_pos = spec.get("goal_pos")
        
        # Placeholder for actual value function inference
        # In practice: return self.value_model(obs, goal_encoding).item()
        
        # For demonstration, if no model is provided, fall back to distance
        if self.value_model is None:
            # Fallback for testing if no model loaded
            return 0.0 
            
        # Mock inference (should be replaced with actual model call)
        # return self.value_model(state, goal_pos).item()
        return 0.0
