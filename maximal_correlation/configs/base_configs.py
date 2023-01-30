from dataclasses import dataclass


@dataclass
class TrainerConfig:
    model_dir: str = ""

    distributed: bool = False
    world_size: int = 2

    log_every: int = 25
    save_every: int = 1000
    max_steps: int = 1_000_000

    learning_rate: float = 2e-4
    train_fraction: float = 0.8
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class KLLossConfig:
    min_logvar: float = -30.0
    max_logvar: float = 20.0


@dataclass
class HScoreConfig:
    feature_dim: int = 192