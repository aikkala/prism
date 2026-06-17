import numpy as np
from prism.reward.base_reward import BaseReward
from prism.utils.trajectory import Trajectory
from prism.tasks.base_task import BaseTask
from typing import Any

class MiniGridDenseReward(BaseReward):
    def compute(self, trajectory: Trajectory, task: BaseTask) -> float:
        """Trajectory-level reward: max dense reward achieved."""
        if not trajectory.states:
            return 0.0
        
        dense_rewards = [self.compute_dense(state, task) for state in trajectory.states]
        return max(dense_rewards)

    def compute_dense(self, state: Any, task: BaseTask) -> float:
        """Dense reward: negative Manhattan distance to goal."""
        spec = task.get_spec()
        goal_pos = spec.get("goal_pos")
        if goal_pos is None:
            return 0.0
        
        # We need to extract agent position from the state
        # state is MiniGrid environment object (from get_state)
        agent_pos = state.agent_pos
        
        distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        return -float(distance)
