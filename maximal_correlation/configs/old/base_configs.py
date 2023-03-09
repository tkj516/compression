from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    model_dir: str = ""

    distributed: bool = False
    world_size: int = 2
    num_workers: int = 2

    log_every: int = 50
    save_every: int = 5000
    validate_every: int = 1000
    max_steps: int = 1_000_000
    hscore_start: int = 5_000

    batch_size: int = 32
    train_fraction: float = 0.8

    clip_max_norm: float = 0.0
    joint_optimization: bool = False
    hscore_freq: int = 1
