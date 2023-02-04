import os
import math
import dacite

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

from models.svdc import SVDC
from losses.hscore import NegativeHScore
from ml_collections import ConfigDict
from dataset import ImageNetTrain
from configs.base_configs import TrainerConfig
from maximal_correlation.utils.class_builder import ClassBuilder


def get_train_val_dataset(dataset: Dataset, train_fraction: float):
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_fraction, 1 - train_fraction], 
        generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


@dataclass
class AdamConfig:
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    amsgrad: bool = False

OPTIMIZER_REGISTER = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}
optimizer_builder = ClassBuilder(OPTIMIZER_REGISTER, AdamConfig)


class Learner:
    def __init__(
        self,
        model: nn.Module,
        cfg: ConfigDict,
        rank: int
    ):
        # Store some important variables
        self.rank = rank
        self.cfg = cfg
        self.step = 0
        
        self.trainer_config = dacite.from_dict(
            TrainerConfig, cfg.trainer_config)
        self.build_dataloaders()

        # Store the model
        self.model = model

        # Instantiate the optimizers
        self.optimizer_e = optimizer_builder.build_class(
            params=self.model.encoder.parameters(),
            config=cfg.optimizer_e_config,
        )
        self.optimizer_ed = optimizer_builder.build_class(
            params=set(p for n, p in self.model.named_parameters() if not n.endswith(".quantiles")),
            config=cfg.optimizer_ed_config
        )
        self.optimizer_aux = optimizer_builder.build_class(
            params=set(p for n, p in self.model.named_parameters() if n.endswith(".quantiles")),
            config=cfg.optimizer_aux_config,
        )

        self.hscore = NegativeHScore(
            feature_dim=self.model.encoder_config.out_channels
        )
        self.rd_lambda = cfg.rd_lambda

        # Insantiate a Tensorboard summary writer
        self.writer = SummaryWriter(self.trainer_config.model_dir)

    @property
    def is_master(self):
        return self.rank == 0

    def build_dataloaders(self):
        self.dataset = ImageNetTrain()
        self.train_dataset, self.val_dataset = get_train_val_dataset(
            self.dataset, self.trainer_config.train_fraction)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=not self.trainer_config.distributed,
            num_workers=self.trainer_config.num_workers if self.trainer_config.distributed else 0,
            sampler=DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer_config.world_size,
                rank=self.rank) if self.trainer_config.distributed else None,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=not self.trainer_config.distributed,
            num_workers=self.trainer_config.num_workers if self.trainer_config.distributed else 0,
            pin_memory=True,
        )

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer_e': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer_e.state_dict().items()},
            'optimizer_ed': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer_ed.state_dict().items()},
            'optimizer_aux': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer_aux.state_dict().items()},
            'cfg': self.cfg.to_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer_e.load_state_dict(state_dict['optimizer_e'])
        self.optimizer_ed.load_state_dict(state_dict['optimizer_ed'])
        self.optimizer_aux.load_state_dict(state_dict['optimizer_aux'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.trainer_config.model_dir}/{save_basename}'
        link_name = f'{self.trainer_config.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(
                f'{self.trainer_config.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def hscore_step(self) -> bool:
        if self.step >= self.trainer_config.hscore_start and self.step % 2 == 0:
            return True

    def train(self):
        device = next(self.model.parameters()).device
        while True:
            for _, inputs in enumerate(
                tqdm(
                    self.train_dataloader,
                    desc=f"Training ({self.step} / {self.trainer_config.max_steps})"
                )
            ):
                x = inputs["image"].to(device)
                loss = self.train_step(x, logging_rank=self.rank == 0)

                # Check for NaNs
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at step {self.step}.')

                if self.is_master:
                    if self.step > 0 and self.step % self.trainer_config.validate_every == 0:
                        self.validate()
                    if self.step % self.trainer_config.save_every == 0:
                        self.save_to_checkpoint()

                if self.trainer_config.distributed:
                    dist.barrier()

                self.step += 1

                if self.step == self.trainer_config.max_steps:
                    if self.is_master and self.trainer_config.distributed:
                        self.save_to_checkpoint()
                        print("Ending training...")
                    dist.barrier()
                    exit(0)

    def train_step(self, x: torch.Tensor, logging_rank: bool = False):
        if self.hscore_step():
            # Optimize the encoder and don't optimize the entropy model
            # or the decoder
            optimizer = self.optimizer_e
            optimizer.zero_grad()

            # 1. Get two augmented inputs
            augmented = self.model.augment(x, num_samples=2)
            z0, z1 = torch.chunk(augmented, chunks=2, dim=0)
            z0, z1 = z0.mean(0), z1.mean(0)

            # 2. Encode the augmented inputs
            f_z = self.model.encode(torch.concatenate([z0, z1], dim=0))
            f_z0, f_z1 = torch.chunk(f_z, chunks=2, dim=0)

            # 3. Compute the negative hscore loss between the encodings
            b, c, h, w = f_z0.shape
            phi = f_z0.permute(0, 2, 3, 1).reshape(b * h * w, c)
            psi = f_z1.permute(0, 2, 3, 1).reshape(b * h * w, c)
            loss = self.hscore(phi, psi, buffer_psi=None)

            loss.backward()
            optimizer.step()

            if logging_rank and self.step % self.trainer_config.log_every == 0:
                self.writer.add_scalar('train/hscore_loss', loss, self.step)
        else:
            optimizer = self.optimizer_ed
            aux_optimizer = self.optimizer_aux

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            outputs = self.model(x, num_samples=8)
            recon, likelihoods = outputs["x_hat"], outputs["y_likelihoods"]

            b, _, h, w = x.shape
            num_pixels = b * h * w

            bpp_loss = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            mse_loss = F.mse_loss(x, recon)
            rd_loss = bpp_loss + self.rd_lambda * (255.0 ** 2) * mse_loss

            rd_loss.backward()
            optimizer.step()

            aux_loss = self.model.entropy_bottleneck.loss()
            aux_loss.backward()
            aux_optimizer.step()

            loss = rd_loss + aux_loss
            
            if logging_rank and self.step % self.trainer_config.log_every == 0:
                self.writer.add_scalar('train/bpp', bpp_loss, self.step)
                self.writer.add_scalar('train/mse', mse_loss, self.step)
                self.writer.add_scalar('train/aux_loss', aux_loss, self.step)
                self.writer.add_scalar('train/rd_loss', rd_loss, self.step)
                self.writer.add_scalar('train/psnr', -10 / math.log(10) * torch.log(mse_loss), self.step)
                self.writer.add_images('train/gt', x, self.step)
                self.writer.add_images('train/recon', recon, self.step)
        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        bpp = 0
        mse = 0
        aux = 0
        loss = 0
        for inputs in tqdm(self.val_dataloader, desc=f"Running validation after step {self.step}"):
            x = inputs["image"].to(device)
            # Use the underlying module to get the losses
            if self.trainer_config.distributed:
                outputs = self.model.module(x, num_samples=8)
            else:
                outputs = self.model(x, num_samples=8)
            recon, likelihoods = outputs["x_hat"], outputs["y_likelihoods"]

            b, _, h, w = x.shape
            num_pixels = b * h * w

            bpp_loss = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            mse_loss = F.mse_loss(x, recon)
            rd_loss = bpp_loss + self.rd_lambda * 255.0 ** 2 * mse_loss
            aux_loss = self.model.entropy_bottleneck.loss()

            bpp += bpp_loss
            mse += mse_loss
            aux += aux_loss
            loss += rd_loss
        bpp = bpp / len(self.val_dataset)
        mse = mse / len(self.val_dataset)
        aux = aux / len(self.val_dataset)
        loss = loss / len(self.val_dataset)

        self.writer.add_scalar('val/bpp', bpp, self.step)
        self.writer.add_scalar('val/mse', mse, self.step)
        self.writer.add_scalar('val/aux_loss', aux, self.step)
        self.writer.add_scalar('val/rd_loss', loss, self.step)
        self.writer.add_scalar('val/psnr', -10 / math.log(10) * torch.log(mse), self.step)
        self.writer.add_images('val/gt', x, self.step)
        self.writer.add_images('val/recon', recon, self.step)
        self.model.train()

        return loss


def _train_impl(rank: int, model: nn.Module, cfg: ConfigDict):
    torch.backends.cudnn.benchmark = True

    learner = Learner(model, cfg, rank)
    learner.restore_from_checkpoint()
    learner.train()


def train(cfg: ConfigDict):
    """Training on a single GPU."""
    model = SVDC(**cfg.model_config).cuda()
    _train_impl(0, model, cfg)


def init_distributed(rank: int, world_size: int, port: str):
    """Initialize distributed training on multiple GPUs."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=rank, world_size=world_size)


def train_distributed(rank: int, world_size: int, port, cfg: ConfigDict):
    """Training on multiple GPUs."""
    init_distributed(rank, world_size, port)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    model = SVDC(**cfg.model_config).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    _train_impl(rank, model, cfg)
