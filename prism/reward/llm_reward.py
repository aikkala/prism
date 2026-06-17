import json
import requests
import re
from typing import Any, List
from prism.reward.base_reward import BaseReward
from prism.utils.trajectory import Trajectory
from prism.tasks.base_task import BaseTask
from prism.utils.config import LLMConfig
from prism.utils.map_desc import describe_minigrid_map

class LLMReward(BaseReward):
    """
    Reward module that uses a local LLM to evaluate full trajectories.
    """
    def __init__(self, config: LLMConfig):
        self.api_url = config.api_url
        self.model = config.model

    def compute(self, trajectory: Trajectory, task: BaseTask) -> float:
        """
        Trajectory-level reward: LLM evaluates the full trajectory 
        given the map description, task, and path.
        """
        if not trajectory.states:
            return 0.0
        
        # 1. Map Description (using the first state as reference for the layout)
        initial_env_state = trajectory.states[0]
        map_desc = describe_minigrid_map(initial_env_state)
        
        # 2. Task Description
        task_spec = task.get_spec()
        goal_pos = task_spec.get("goal_pos", "unknown")
        task_desc = f"Navigate to the goal at {goal_pos}."
        
        # 3. Path Description
        path = []
        for state in trajectory.states:
            pos = getattr(state, 'agent_pos', 'unknown')
            path.append(f"({pos[0]}, {pos[1]})")
        
        prompt = (
            f"### Environment Description\n{map_desc}\n"
            f"### Task\n{task_desc}\n"
            f"### Trajectory Path\n{' -> '.join(path)}\n\n"
            "### Instructions\n"
            "Evaluate the trajectory above. Did the agent move towards the goal? "
            "Rate the overall quality and success of this trajectory on a scale from 0.0 (failure) to 1.0 (perfect success).\n"
            "Return ONLY the numerical score."
        )
        
        return self._query_llm(prompt)

    def compute_dense(self, state: Any, task: BaseTask) -> float:
        """
        Dense reward is no longer estimated via LLM per step to save resources.
        Returns 0.0 as MCTS will now rely on trajectory-level LLM evaluation 
        at the end of rollouts/simulations.
        """
        return 0.0

    def _query_llm(self, prompt: str) -> float:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "0.0").strip()
            
            match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
            if match:
                return float(match.group())
            return 0.0
        except Exception:
            return 0.0
