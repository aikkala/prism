import random
from typing import Dict, List
from prism.tasks.base_task import BaseTask

class MiniGridTask(BaseTask):
    def __init__(self, goal_pos: tuple, env_id: str = "MiniGrid-Empty-8x8-v0"):
        self.goal_pos = goal_pos
        self.env_id = env_id

    def get_spec(self) -> Dict:
        return {"goal_pos": self.goal_pos}

    def get_env_config(self) -> Dict:
        return {"env_id": self.env_id}

class TaskSampler:
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    def sample(self) -> MiniGridTask:
        # Sample a random goal position that is not the start (1,1)
        # Assuming MiniGrid-Empty start is usually (1,1)
        while True:
            x = random.randint(1, self.grid_size - 2)
            y = random.randint(1, self.grid_size - 2)
            if (x, y) != (1, 1):
                break
        return MiniGridTask(goal_pos=(x, y))

    def sample_batch(self, batch_size: int) -> List[MiniGridTask]:
        return [self.sample() for _ in range(batch_size)]
