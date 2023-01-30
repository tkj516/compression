import os
from typing import Dict
import dacite

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vae import GaussianVAE
from losses.hscore import NegativeHScore
from losses.kl import KLLoss
from dataclasses import dataclass
from ml_collections import ConfigDict
from dataset import ImageNetTrain
from configs.base_configs import TrainerConfig


def get_train_val_dataset(dataset: Dataset, train_fraction: float):
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_fraction, 1 - train_fraction], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


class Learner:
    def __init__(
        self,
        model: nn.Module,
        cfg: ConfigDict,
        rank: int
    ):
        # Store some import variables
        self.rank = rank
        self.cfg = cfg
        self.trainer_config = dacite.from_dict(TrainerConfig, cfg.trainer_config)
        self.build_dataloaders()

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.trainer_config.learning_rate)
        self.step = 0

        # Define the loss functions
        self.kl = KLLoss(**cfg.vae_loss_config)
        self.hscore = NegativeHScore(**cfg.hscore_loss_config)
        self.l1 = nn.L1Loss()

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
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'cfg': self.cfg.to_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
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
                    if self.step % self.trainer_config.validate_every == 0:
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
        mean, logvar = self.model.encode(x)

        if self.step % 2 == 0:
            # Optimize the VAE
            sample = self.model.sample(mean, logvar)
            recon = self.model.decode(sample)
            kl_loss = self.kl(mean, logvar)
            l1_loss = self.l1(recon, x)
            loss = kl_loss + l1_loss
            if logging_rank and self.step % self.trainer_config.log_every == 0:
                self.writer.add_scalar('train/kl_loss', kl_loss)
                self.writer.add_scalar('train/l1_loss', l1_loss)
                self.writer.add_scalar('train/vae_loss', loss)
        else:
            # Optimize the HScore
            sample_0 = self.model.sample(mean, logvar)
            sample_1 = self.model.sample(mean, logvar)
            loss = self.hscore(sample_0, sample_1)
            if logging_rank and self.step % self.trainer_config.log_every == 0:
                self.writer.add_scalar('train/hscore', loss)

        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        kl_loss = 0
        l1_loss = 0
        loss = 0
        for inputs in tqdm(self.val_dataloader, desc=f"Running validation after step {self.step}"):
            x = inputs["image"].to(device)
            # Use the underlying module to get the losses
            if self.distributed:
                mean, logvar, recon = self.model.module(x)
            else:
                mean, logvar, recon = self.model(x)
            loss_kl = self.kl(mean, logvar)
            loss_l1 = self.l1(recon, x)
            kl_loss += loss_kl
            l1_loss += loss_l1
            loss += (loss_kl + loss_l1)
        kl_loss = kl_loss / len(self.val_dataset)
        l1_loss = l1_loss / len(self.val_dataset)
        loss = loss / len(self.val_dataset)

        self.writer.add_scalar('val/kl_loss', kl_loss, self.step)
        self.writer.add_scalar('val/l1_loss', l1_loss, self.step)
        self.writer.add_scalar('val/vae_loss', )
        self.model.train()

        return loss


def _train_impl(rank: int, model: nn.Module, cfg: ConfigDict):
    torch.backends.cudnn.benchmark = True

    learner = Learner(model, cfg, rank)
    learner.restore_from_checkpoint()
    learner.train()


def train(cfg: ConfigDict):
    """Training on a single GPU."""
    model = GaussianVAE(**cfg.model_config).cuda()
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
    model = GaussianVAE(**cfg.model_config).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    _train_impl(rank, model, cfg)
