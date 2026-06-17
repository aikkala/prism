import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from prism.environments.base_env import BaseEnv
from typing import Any, Tuple

class MiniGridEnv(BaseEnv):
    def __init__(self, env_id: str = "MiniGrid-Empty-8x8-v0", **kwargs):
        self.env = gym.make(env_id, **kwargs)
        self.env = FlatObsWrapper(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, task: Optional[Any] = None) -> Any:
        obs, _ = self.env.reset()
        if task:
            # Set goal position from task spec
            spec = task.get_spec()
            goal_pos = spec.get("goal_pos")
            if goal_pos:
                from minigrid.core.world_object import Goal
                # Remove old goal if any
                grid = self.env.unwrapped.grid
                for i in range(grid.width):
                    for j in range(grid.height):
                        obj = grid.get(i, j)
                        if isinstance(obj, Goal):
                            grid.set(i, j, None)
                # Place new goal
                grid.set(goal_pos[0], goal_pos[1], Goal())
                # Re-generate observation after grid change
                obs = self.env.observation(self.env.gen_obs())
        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, float(reward), done, info

    def get_state(self) -> Any:
        """Returns the internal state of the MiniGrid environment."""
        # For MiniGrid, we can often just return the internal grid and agent state
        # However, a more robust way for Gymnasium/MiniGrid is to use deepcopy or similar
        # if the environment supports it. 
        # For MCTS purposes, we need a way to restore.
        # This implementation might need to be more specific to MiniGrid's internal state.
        import copy
        return copy.deepcopy(self.env.unwrapped)

    def set_state(self, state: Any):
        """Sets the internal state of the MiniGrid environment."""
        import copy
        self.env.unwrapped.__dict__.update(state.__dict__)

    def process_obs(self, obs: Any) -> Any:
        """Processes a raw observation using the FlatObsWrapper."""
        return self.env.observation(obs)
