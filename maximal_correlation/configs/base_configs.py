from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerConfig:
    model_dir: str = ""

    distributed: bool = False
    world_size: int = 2

    log_every: int = 50
    save_every: int = 1000
    validate_every: int = 1000
    max_steps: int = 1_000_000

    learning_rate: float = 2e-4
    train_fraction: float = 0.8
    batch_size: int = 16
    num_workers: int = 2


@dataclass
class KLLossConfig:
    min_logvar: Optional[float] = None
    max_logvar: Optional[float] = None
    kld_weight: float = 0.00025


@dataclass
class HScoreConfig:
    feature_dim: int = 192