from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

class BaseEnv(ABC):
    @abstractmethod
    def reset(self, task: Optional[Any] = None) -> Any:
        """Resets the environment for a specific task and returns initial observation."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Steps the environment and returns (obs, reward, done, info)."""
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """Returns the internal state of the environment."""
        pass

    @abstractmethod
    def process_obs(self, obs: Any) -> Any:
        """Processes a raw observation into the format expected by the policy."""
        pass
