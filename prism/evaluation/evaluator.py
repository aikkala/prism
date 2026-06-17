import copy
from typing import Dict, Any
from prism.agent.base_agent import BaseAgent
from prism.tasks.minigrid_tasks import TaskSampler, MiniGridTask
from prism.environments.minigrid_env import MiniGridEnv
from prism.utils.config import Config

class Evaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, agent: BaseAgent, task_sampler: TaskSampler) -> Dict[str, float]:
        """Runs evaluation of the current meta-agent on a set of tasks."""
        total_return = 0.0
        success_count = 0
        n_tasks = self.config.evaluation.held_out_tasks
        
        for _ in range(n_tasks):
            task = task_sampler.sample()
            
            # Clone agent for adaptation (inner loop is not run here for simplicity,
            # but in a real meta-learning setup, you'd adapt the agent to the task first)
            # For this evaluator, we'll just test the meta-agent directly or adapt it briefly.
            eval_agent = copy.deepcopy(agent)
            
            # Create environment for this task
            env_config = task.get_env_config()
            env = MiniGridEnv(env_id=env_config['env_id'])
            
            # Execute policy in the environment
            obs = env.reset()
            episode_return = 0.0
            done = False
            step_count = 0
            
            while not done and step_count < 100: # Max steps
                action = eval_agent.act(obs)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                step_count += 1
                
            total_return += episode_return
            if episode_return > 0: # Assuming positive reward for success in MiniGrid
                success_count += 1
                
        return {
            "mean_return": total_return / n_tasks,
            "success_rate": success_count / n_tasks
        }
