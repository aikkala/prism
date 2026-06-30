from dataclasses import dataclass, field
from typing import List, Any
import torch

@dataclass
class Trajectory:
    states: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    done: bool = False
    total_reward: float = 0.0
    observations: List[Any] = field(default_factory=list)

    def length(self) -> int:
        return len(self.actions)

    def to_tensors(self, device='cpu'):
        """Converts to PyTorch tensors for training."""
        # Use observations if available, otherwise states
        states_to_convert = self.observations if (hasattr(self, 'observations') and self.observations) else self.states
        states_tensor = torch.tensor(states_to_convert, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        return states_tensor, actions_tensor, rewards_tensor
