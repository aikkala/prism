import torch
from prism.utils.config import Config
from prism.utils.logging import Logger
from prism.environments.minigrid_env import MiniGridEnv
from prism.world_model.perfect_world_model import PerfectWorldModel
from prism.reward.llm_reward import LLMReward
from prism.planner.mcts import MCTSPlanner
from prism.agent.policy import Policy
from prism.agent.meta_agent import MetaAgent
from prism.imitation.behavioral_cloning import BehavioralCloning
from prism.meta_learning.reptile import ReptileMetaLearner
from prism.tasks.minigrid_tasks import TaskSampler
from prism.training.meta_trainer import MetaTrainer
from prism.evaluation.evaluator import Evaluator

def main():
    # Optimize CPU operations for small neural network models
    torch.set_num_threads(1)
    
    # 1. Load config
    config = Config()
    
    # 2. Setup logging
    logger = Logger(use_wandb=config.use_wandb, project_name=config.project_name)
    
    # 3. Instantiate environment and world model
    # We use one instance for the world model to interact with
    env = MiniGridEnv()
    world_model = PerfectWorldModel(env)
    
    # 4. Instantiate reward function and planner
    # Using local LLM reward (e.g. Ollama)
    reward_fn = LLMReward(config.llm)
    planner = MCTSPlanner(config.mcts)
    
    # 5. Instantiate agent
    # Get observation and action dimensions from environment
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(input_dim=obs_dim, action_dim=action_dim)
    agent = MetaAgent(policy)
    
    # 6. Instantiate learners
    imitation_learner = BehavioralCloning(config.bc)
    meta_learner = ReptileMetaLearner(config.reptile)
    
    # 7. Instantiate tasks and evaluator
    task_sampler = TaskSampler()
    evaluator = Evaluator(config)
    
    # 8. Instantiate and run trainer
    trainer = MetaTrainer(
        config=config,
        agent=agent,
        env=env,
        world_model=world_model,
        planner=planner,
        reward_fn=reward_fn,
        imitation_learner=imitation_learner,
        meta_learner=meta_learner,
        task_sampler=task_sampler,
        evaluator=evaluator,
        logger=logger
    )
    
    logger.logger.info("Starting PRISM training with LLM reward...")
    trainer.run()

if __name__ == "__main__":
    main()
