from typing import Dict, Any
from prism.agent.base_agent import BaseAgent
from prism.tasks.base_task import BaseTask
from prism.world_model.base_world_model import BaseWorldModel
from prism.planner.base_planner import BasePlanner
from prism.reward.base_reward import BaseReward
from prism.imitation.base_imitation import BaseImitationLearner
from prism.utils.trajectory import Trajectory

def run_inner_loop(
    agent: BaseAgent,
    task: BaseTask,
    world_model: BaseWorldModel,
    planner: BasePlanner,
    reward_fn: BaseReward,
    imitation_learner: BaseImitationLearner,
    env: Any # Pass real environment for execution
) -> Dict[str, Any]:
    """Executes one full inner loop: MCTS planning, BC update, and real execution."""

    # 1. Reset world model for this task
    initial_state = world_model.reset(task)

    # 2. Run MCTS to get trajectories
    trajectories = planner.plan(initial_state, task, world_model, reward_fn, policy=agent)

    if not trajectories:
        return {'loss': 0.0, 'updated_params': agent.get_params(), 'trajectories': [], 'eval_reward': 0.0}

    # 3. Run behavioral cloning update on the agent (Adaptation)
    loss = imitation_learner.update(agent, trajectories)

    # 4. Execute the UPDATED policy in the real environment
    obs = env.reset(task)
    done = False
    total_reward = 0.0
    step_count = 0
    query_states = [obs]
    query_actions = []

    while not done and step_count < 100:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        query_states.append(obs)
        query_actions.append(action)
        step_count += 1

    # 5. Return results including real-world performance
    return {
        'loss': loss,
        'updated_params': agent.get_params(),
        'trajectories': trajectories,
        'query_trajectory': Trajectory(states=query_states, actions=query_actions),
        'eval_reward': total_reward
    }


