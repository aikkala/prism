from dataclasses import dataclass, field
from typing import Dict, Any
import os


@dataclass
class MCTSConfig:
    n_simulations: int = 100
    max_depth: int = 50
    exploration_constant: float = 1.414
    n_rollouts_returned: int = 5
    epsilon: float = 0.1 # Probability of taking a random action during rollout

@dataclass
class ReptileConfig:
    meta_lr: float = 0.001
    n_inner_steps: int = 5

@dataclass
class BCConfig:
    n_gradient_steps: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32

@dataclass
class TrainingConfig:
    n_meta_iterations: int = 1000
    tasks_per_batch: int = 4
    eval_interval: int = 50

@dataclass
class EvaluationConfig:
    n_eval_episodes: int = 10
    held_out_tasks: int = 20

@dataclass
class LLMConfig:
    model: str = os.environ.get("PRISM_LLM_MODEL", "gemma4:12b")
    api_url: str = "http://localhost:11434/api/generate"

@dataclass
class Config:
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    reptile: ReptileConfig = field(default_factory=ReptileConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    project_name: str = "prism"
    use_wandb: bool = False
