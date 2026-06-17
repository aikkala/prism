import torch
from typing import List, Dict, Any
from prism.meta_learning.base_meta_learner import BaseMetaLearner
from prism.agent.base_agent import BaseAgent
from prism.utils.config import ReptileConfig

class ReptileMetaLearner(BaseMetaLearner):
    def __init__(self, config: ReptileConfig):
        self.config = config

    def meta_update(self, agent: BaseAgent, task_results: List[Dict[str, Any]]):
        """
        Reptile update: theta = theta + epsilon * avg(theta_task - theta)
        where theta is the initial meta-parameters.
        """
        meta_params = agent.get_params()
        
        # Calculate average direction
        avg_grad = {k: torch.zeros_like(v) for k, v in meta_params.items()}
        
        for result in task_results:
            updated_params = result['updated_params']
            for k in meta_params:
                avg_grad[k] += (updated_params[k] - meta_params[k]) / len(task_results)
                
        # Apply update
        new_meta_params = {k: meta_params[k] + self.config.meta_lr * avg_grad[k] for k in meta_params}
        agent.set_params(new_meta_params)
