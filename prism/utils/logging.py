import logging
from typing import Any, Dict

class Logger:
    def __init__(self, use_wandb: bool = False, project_name: str = "prism"):
        self.logger = logging.getLogger("prism")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb
            wandb.init(project=project_name)

    def log_scalar(self, key: str, value: Any, step: int):
        self.logger.info(f"Step {step} - {key}: {value}")
        if self.use_wandb:
            import wandb
            wandb.log({key: value}, step=step)

    def log_trajectory(self, trajectory: Any, step: int):
        # Placeholder for trajectory logging logic
        self.logger.info(f"Step {step} - Logged trajectory of length {trajectory.length()}")
        if self.use_wandb:
            # Optionally log trajectory data to wandb
            pass

    def log_config(self, config: Any):
        self.logger.info(f"Config: {config}")
        if self.use_wandb:
            import wandb
            wandb.config.update(config)
