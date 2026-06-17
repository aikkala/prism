import torch
from typing import Any, List, Dict
from prism.agent.base_agent import BaseAgent
from prism.agent.policy import Policy
from prism.utils.trajectory import Trajectory

class MetaAgent(BaseAgent):
    def __init__(self, policy: Policy, device: str = 'cpu'):
        self.policy = policy
        self.device = device

    def act(self, obs: Any) -> Any:
        self.policy.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.policy(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()
        return action

    def update(self, trajectories: List[Trajectory]):
        # This is typically handled by the imitation learner
        pass

    def get_params(self) -> Dict[str, Any]:
        return {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}

    def set_params(self, params: Dict[str, Any]):
        self.policy.load_state_dict(params)
