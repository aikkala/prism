import copy
from typing import List, Dict, Any
from prism.environments.base_env import BaseEnv
from prism.agent.base_agent import BaseAgent
from prism.world_model.base_world_model import BaseWorldModel
from prism.planner.base_planner import BasePlanner
from prism.reward.base_reward import BaseReward
from prism.imitation.base_imitation import BaseImitationLearner
from prism.meta_learning.base_meta_learner import BaseMetaLearner
from prism.tasks.minigrid_tasks import TaskSampler
from prism.training.inner_loop import run_inner_loop
from prism.utils.logging import Logger
from prism.utils.config import Config
from prism.evaluation.evaluator import Evaluator

class MetaTrainer:
    def __init__(
        self,
        config: Config,
        agent: BaseAgent,
        env: BaseEnv, # Added real environment
        world_model: BaseWorldModel,
        planner: BasePlanner,
        reward_fn: BaseReward,
        imitation_learner: BaseImitationLearner,
        meta_learner: BaseMetaLearner,
        task_sampler: TaskSampler,
        evaluator: Evaluator,
        logger: Logger
    ):
        self.config = config
        self.agent = agent
        self.env = env
        self.world_model = world_model
        self.planner = planner
        self.reward_fn = reward_fn
        self.imitation_learner = imitation_learner
        self.meta_learner = meta_learner
        self.task_sampler = task_sampler
        self.evaluator = evaluator
        self.logger = logger

    def run(self):
        self.logger.log_config(self.config)
        
        for iteration in range(self.config.training.n_meta_iterations):
            # Sample a batch of tasks
            tasks = self.task_sampler.sample_batch(self.config.training.tasks_per_batch)
            
            task_results = []
            total_loss = 0.0
            total_eval_reward = 0.0
            
            for task in tasks:
                # Clone the agent for the inner loop
                task_agent = copy.deepcopy(self.agent)
                
                # Run inner loop (including real env execution)
                result = run_inner_loop(
                    task_agent, task, self.world_model, self.planner, 
                    self.reward_fn, self.imitation_learner, self.env
                )
                task_results.append(result)
                total_loss += result['loss']
                total_eval_reward += result.get('eval_reward', 0.0)
                
            # Perform meta-update
            self.meta_learner.meta_update(self.agent, task_results)
            
            # Logging
            avg_loss = total_loss / len(tasks)
            avg_eval_reward = total_eval_reward / len(tasks)
            self.logger.log_scalar("meta_loss", avg_loss, iteration)
            self.logger.log_scalar("train/eval_reward", avg_eval_reward, iteration)
            
            # Evaluation
            if iteration % self.config.training.eval_interval == 0:
                eval_metrics = self.evaluator.evaluate(self.agent, self.task_sampler)
                for k, v in eval_metrics.items():
                    self.logger.log_scalar(f"eval/{k}", v, iteration)
