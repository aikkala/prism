from typing import Any, Tuple
from prism.world_model.base_world_model import BaseWorldModel
from prism.environments.base_env import BaseEnv
from prism.tasks.base_task import BaseTask

class PerfectWorldModel(BaseWorldModel):
    def __init__(self, env: BaseEnv):
        self.env = env

    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        # Save current env state
        current_state = self.env.get_state()
        
        # Set env to the state we want to transition from
        self.env.set_state(state)
        
        # Take action
        obs, reward, done, _ = self.env.step(action)
        
        # Get next state
        next_state = self.env.get_state()
        
        # Restore original env state
        self.env.set_state(current_state)
        
        return next_state, reward, done

    def reset(self, task: BaseTask) -> Any:
        # Use the task to correctly configure the environment
        obs = self.env.reset(task)
        return self.env.get_state()
