# PRISM: Planning via Rollout Imagination for Sample-efficient Meta-learning

## Project Overview
PRISM is a modular meta-learning framework designed for rapid task adaptation. It leverages Monte Carlo Tree Search (MCTS) over a world model to discover expert-level trajectories, uses these for Behavioral Cloning (inner loop), and optimizes initial parameters via Reptile (outer loop).

## Core Architecture & Conventions

### Modularity
- **Strict Dependency Injection:** Concrete classes are ONLY instantiated in `main.py`. 
- **Abstract Interfaces:** All inter-module communication must happen via the abstract base classes defined in `prism/*/base_*.py`.
- **Environment:** Primarily targets `MiniGrid` (Gymnasium).

### Key Modules
- **`environments/`**: `MiniGridEnv` handles task-aware resets and goal placement.
- **`world_model/`**: `PerfectWorldModel` uses env state-capturing for zero-shot imagination.
- **`reward/`**: Uses `LLMReward` (via local Ollama API) to evaluate **full trajectories** rather than single steps.
- **`planner/`**: `MCTSPlanner` implements policy-guided search.
- **`agent/`**: `MetaAgent` wraps a 2-headed neural `Policy` (actor/critic).
- **`training/`**: 
    - `run_inner_loop`: Planning -> BC Adaptation -> Real-world Execution.
    - `MetaTrainer`: Orchestrates meta-iterations across task batches.

## Technical Details for Continuation

### MCTS Implementation
- **Policy-Guided:** Uses the agent's current policy for epsilon-greedy selection, expansion, and rollouts.
- **Trajectory Evaluation:** The LLM evaluates the *entire* path from root to rollout end in a single call per simulation.
- **Observation Processing:** MCTS must use `world_model.env.process_obs()` to ensure input dimensions (2835 for MiniGrid) match the policy expectations.

### Reward Prompting
- The LLM prompt includes a text-based **Map Description**, the **Task**, and the **Trajectory Path** (agent coordinates).
- Expected LLM response is a single numerical score (0.0 to 1.0).

### Environment & Dependencies
- **Conda Env:** `prism` (Python 3.11).
- **Packages:** `torch`, `gymnasium`, `minigrid`, `numpy`, `requests`.
- **Local LLM:** Ollama (default model: `llama3`) served at `http://localhost:11434`.

## Active Workflow
- The training loop is currently grounded in real-environment execution.
- Performance is tracked via `train/eval_reward` in the logs.
