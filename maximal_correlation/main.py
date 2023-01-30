import dacite
import socketserver
from ml_collections import config_flags, ConfigDict
from torch.cuda import device_count
from torch.multiprocessing import spawn
from dataclasses import dataclass
from learner import train, train_distributed
from absl import app
from configs.base_configs import TrainerConfig


CONFIG = config_flags.DEFINE_config_file("config")


def _get_free_port():
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def main(_):
    cfg = ConfigDict(CONFIG.value)

    # Setup training
    world_size = device_count()
    trainer_config = dacite.from_dict(TrainerConfig, cfg.trainer_config)
    if trainer_config.distributed and world_size != trainer_config.world_size:
        raise ValueError(
            "Requested world size is not the same as number of visible GPUs.")
    if trainer_config.distributed:
        if world_size < 2:
            raise ValueError(
                f"Distributed training cannot be run on machine with {world_size} device(s).")
        if trainer_config.batch_size % world_size != 0:
            raise ValueError(
                f"Batch size {trainer_config.batch_size} is not evenly divisble by # GPUs = {world_size}.")
        cfg.trainer_config.batch_size = trainer_config.batch_size // world_size
        port = _get_free_port()
        spawn(train_distributed, args=(world_size, port, cfg), nprocs=world_size, join=True)
    else:
        train(cfg)


if __name__ == "__main__":
    app.run(main)