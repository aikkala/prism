from abc import ABC, abstractmethod
from typing import Any, Tuple
from prism.tasks.base_task import BaseTask

class BaseWorldModel(ABC):
    @abstractmethod
    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        """Predicts next state, reward, and done signal."""
        pass

    @abstractmethod
    def reset(self, task: BaseTask) -> Any:
        """Returns the initial state for a given task."""
        pass
