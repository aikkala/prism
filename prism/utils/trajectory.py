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

    def length(self) -> int:
        return len(self.actions)

    def to_tensors(self, device='cpu'):
        """Converts to PyTorch tensors for training."""
        # Assuming states are numpy arrays or similar that can be converted to tensors
        states_tensor = torch.tensor(self.states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        return states_tensor, actions_tensor, rewards_tensor
