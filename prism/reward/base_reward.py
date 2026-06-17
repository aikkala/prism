from abc import ABC, abstractmethod
from typing import Any
from prism.utils.trajectory import Trajectory
from prism.tasks.base_task import BaseTask

class BaseReward(ABC):
    @abstractmethod
    def compute(self, trajectory: Trajectory, task: BaseTask) -> float:
        """Computes trajectory-level reward."""
        pass

    @abstractmethod
    def compute_dense(self, state: Any, task: BaseTask) -> float:
        """Computes dense per-step reward for guiding search."""
        pass
