from abc import ABC, abstractmethod
from typing import Dict

class BaseTask(ABC):
    @abstractmethod
    def get_spec(self) -> Dict:
        """Returns the task specification for reward calculation."""
        pass

    @abstractmethod
    def get_env_config(self) -> Dict:
        """Returns the configuration for environment instantiation."""
        pass
