from abc import ABC, abstractmethod
from typing import List
from prism.utils.trajectory import Trajectory
from prism.agent.base_agent import BaseAgent

class BaseImitationLearner(ABC):
    @abstractmethod
    def update(self, agent: BaseAgent, trajectories: List[Trajectory]) -> float:
        """Performs imitation learning update and returns loss."""
        pass
