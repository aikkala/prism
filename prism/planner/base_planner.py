from abc import ABC, abstractmethod
from typing import List, Any, Optional
from prism.utils.trajectory import Trajectory
from prism.tasks.base_task import BaseTask
from prism.world_model.base_world_model import BaseWorldModel
from prism.reward.base_reward import BaseReward

class BasePlanner(ABC):
    @abstractmethod
    def plan(self, initial_state: Any, task: BaseTask, world_model: BaseWorldModel, reward: BaseReward, policy: Optional[Any] = None) -> List[Trajectory]:
        """Returns a ranked list of trajectories."""
        pass
