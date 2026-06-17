from abc import ABC, abstractmethod
from typing import Any, List, Dict
from prism.utils.trajectory import Trajectory

class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs: Any) -> Any:
        """Returns an action for the given observation."""
        pass

    @abstractmethod
    def update(self, trajectories: List[Trajectory]):
        """Updates the agent based on provided trajectories."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Returns the current agent parameters."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]):
        """Sets the agent parameters."""
        pass
