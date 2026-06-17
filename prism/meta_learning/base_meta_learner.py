from abc import ABC, abstractmethod
from typing import List, Dict, Any
from prism.agent.base_agent import BaseAgent

class BaseMetaLearner(ABC):
    @abstractmethod
    def meta_update(self, agent: BaseAgent, task_results: List[Dict[str, Any]]):
        """Performs outer loop meta-update."""
        pass
