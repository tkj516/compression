import os
import math
import dacite

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

from models.svdc import SVDC
from losses.hscore import NegativeHScore, compute_norms
from ml_collections import ConfigDict
from dataset import ImageNetTrain
from configs.base_configs import TrainerConfig
from compressai.optimizers import net_aux_optimizer
from maximal_correlation.utils.class_builder import ClassBuilder


def get_train_val_dataset(dataset: Dataset, train_fraction: float):
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_fraction, 1 - train_fraction],
        generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


def quantize_image(image):
    image = torch.round(image * 255)
    return image.clamp(0, 255).byte()


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


class MyDistributedDataParallel(DistributedDataParallel):
    def __init__(self, model, **kwargs):
        super(MyDistributedDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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
        if not self.trainer_config.joint_optimization:
            self.optimizer_e = optimizer_builder.build_class(
                params=self.model.encoder.parameters(),
                config=cfg.optimizer_e_config,
            )
        conf = {
            "net": {**{"type": cfg.optimizer_ed_config[0]}, **cfg.optimizer_ed_config[1]},
            "aux": {**{"type": cfg.optimizer_aux_config[0]}, **cfg.optimizer_aux_config[1]},
        }
        optimizer = net_aux_optimizer(self.model, conf)
        self.optimizer_ed, self.optimizer_aux = optimizer["net"], optimizer["aux"]
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_ed, "min")

        self.rd_lambda = cfg.rd_lambda
        self.hscore = NegativeHScore(
            feature_dim=self.model.encoder_config.out_channels
        )
        if self.trainer_config.joint_optimization:
            self.hscore_lambda = cfg.hscore_lambda

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
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer_e": self.optimizer_e.state_dict() if not self.trainer_config.joint_optimization else {},
            "optimizer_ed": self.optimizer_ed.state_dict(),
            "optimizer_aux": self.optimizer_aux.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "cfg":  self.cfg.to_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        if not self.trainer_config.joint_optimization:
            self.optimizer_e.load_state_dict(state_dict['optimizer_e'])
        self.optimizer_ed.load_state_dict(state_dict['optimizer_ed'])
        self.optimizer_aux.load_state_dict(state_dict['optimizer_aux'])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
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
                if self.trainer_config.joint_optimization:
                    loss = self.train_step_joint(x, logging_rank=self.rank == 0)
                else:
                    loss = self.train_step_sep(x, logging_rank=self.rank == 0)

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
            # At the end of the epoch
            loss = self.validate()
            self.lr_scheduler.step(loss)

    def train_step_sep(self, x: torch.Tensor, logging_rank: bool = False):
        if self.step >= self.trainer_config.hscore_start and self.step % 2 == 1:
            # Optimize the encoder and don't optimize the entropy model
            # or the decoder
            self.optimizer_e.zero_grad()

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
            self.optimizer_e.step()

            if logging_rank and self.step % self.trainer_config.log_every == 1:
                self.writer.add_scalar('train/hscore_loss', loss, self.step)
        else:
            self.optimizer_ed.zero_grad()
            self.optimizer_aux.zero_grad()

            outputs = self.model(x, num_samples=8)
            recon, likelihoods = outputs["x_hat"], outputs["y_likelihoods"]

            b, _, h, w = x.shape
            num_pixels = b * h * w

            bpp_loss = torch.log(likelihoods).sum() / \
                (-math.log(2) * num_pixels)
            mse_loss = F.mse_loss(x, recon)
            rd_loss = bpp_loss + self.rd_lambda * (255.0 ** 2) * mse_loss

            rd_loss.backward()
            if self.trainer_config.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), self.trainer_config.clip_max_norm)
            self.optimizer_ed.step()

            aux_loss = self.model.entropy_bottleneck.loss()
            aux_loss.backward()
            self.optimizer_aux.step()

            loss = rd_loss + aux_loss

            if logging_rank and self.step % self.trainer_config.log_every == 0:
                self.writer.add_scalar('train/bpp', bpp_loss, self.step)
                self.writer.add_scalar('train/mse', mse_loss, self.step)
                self.writer.add_scalar('train/aux_loss', aux_loss, self.step)
                self.writer.add_scalar('train/rd_loss', rd_loss, self.step)
                self.writer.add_scalar(
                    'train/psnr', -10 / math.log(10) * torch.log(mse_loss), self.step)
                self.writer.add_images('train/gt', quantize_image(x), self.step)
                self.writer.add_images(
                    'train/recon', quantize_image(recon), self.step)
        return loss

    def train_step_joint(self, x: torch.Tensor, logging_rank: bool = False):
        self.optimizer_ed.zero_grad()
        self.optimizer_aux.zero_grad()

        # 1. Get two augmented inputs
        augmented = self.model.augment(x, num_samples=2)
        x0, x1 = torch.chunk(augmented, chunks=2, dim=0)
        x0, x1 = x0.mean(0), x1.mean(0)

        # 2. Encode the augmented inputs
        f_z = self.model.encode(torch.concatenate([x0, x1], dim=0))
        f_z0, f_z1 = torch.chunk(f_z, chunks=2, dim=0)

        # 3. Compute the negative hscore loss between the encodings
        b, c, h, w = f_z0.shape
        phi = f_z0.permute(0, 2, 3, 1).reshape(b * h * w, c)
        psi = f_z1.permute(0, 2, 3, 1).reshape(b * h * w, c)
        hscore_loss = self.hscore(phi, psi, buffer_psi=None)

        # 4. Quantize the latents
        f_z_hat, likelihoods = self.model.quantize((f_z0 + f_z1) / 2.0)

        # 5. Decoder the images
        recon = self.model.decode(f_z_hat)

        b, _, h, w = x.shape
        num_pixels = b * h * w
        bpp_loss = torch.log(likelihoods).sum() / \
            (-math.log(2) * num_pixels)
        mse_loss = F.mse_loss(x, recon)
        rd_loss = bpp_loss + self.rd_lambda * (255.0 ** 2) * mse_loss

        rd_loss.backward()
        if self.trainer_config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm(
                self.model.parameters(), self.trainer_config.clip_max_norm)
        self.optimizer_ed.step()

        aux_loss = self.model.entropy_bottleneck.loss()
        aux_loss.backward()
        self.optimizer_aux.step()

        loss = rd_loss + aux_loss + self.hscore_lambda * hscore_loss

        if logging_rank and self.step % self.trainer_config.log_every == 0:
            self.writer.add_scalar('train/bpp', bpp_loss, self.step)
            self.writer.add_scalar('train/mse', mse_loss, self.step)
            self.writer.add_scalar('train/aux_loss', aux_loss, self.step)
            self.writer.add_scalar('train/rd_loss', rd_loss, self.step)
            self.writer.add_scalar('train/hscore_loss', hscore_loss, self.step)
            self.writer.add_scalar(
                'train/psnr', -10 / math.log(10) * torch.log(mse_loss), self.step)
            self.writer.add_images('train/gt', quantize_image(x), self.step)
            self.writer.add_images(
                'train/recon', quantize_image(recon), self.step)
        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        bpp = 0
        mse = 0
        aux = 0
        loss = 0
        count = 0
        for inputs in tqdm(self.val_dataloader, desc=f"Running validation after step {self.step}"):
            x = inputs["image"].to(device)
            outputs = self.model(x, num_samples=8)
            recon, likelihoods = outputs["x_hat"], outputs["y_likelihoods"]

            b, _, h, w = x.shape
            num_pixels = b * h * w

            bpp_loss = torch.log(likelihoods).sum() / \
                (-math.log(2) * num_pixels)
            mse_loss = F.mse_loss(x, recon)
            rd_loss = bpp_loss + self.rd_lambda * (255.0 ** 2) * mse_loss
            aux_loss = self.model.entropy_bottleneck.loss()

            bpp += bpp_loss
            mse += mse_loss
            aux += aux_loss
            loss += rd_loss
            count += 1
        bpp = bpp / count
        mse = mse / count
        aux = aux / count
        loss = loss / count

        self.writer.add_scalar('val/bpp', bpp, self.step)
        self.writer.add_scalar('val/mse', mse, self.step)
        self.writer.add_scalar('val/aux_loss', aux, self.step)
        self.writer.add_scalar('val/rd_loss', loss, self.step)
        self.writer.add_scalar('val/psnr', -10 / math.log(10)
                               * torch.log(mse), self.step)
        self.writer.add_images('val/gt', quantize_image(x), self.step)
        self.writer.add_images('val/recon', quantize_image(recon), self.step)
        self.plot_singular_values(x)
        self.model.train()

        return loss

    def plot_singular_values(self, x: torch.Tensor) -> None:
        def encoder(x: torch.Tensor) -> torch.Tensor:
            # 1. Augment the input input image and get multiple samples
            augmented = self.model.augment(x, num_samples=2)
            x_i = [sample.squeeze(0) for sample in torch.chunk(
                augmented, chunks=2, dim=0)]

            # 2. Encode the augmented samples and compute the expected values
            f_z = self.model.encode(torch.concatenate(x_i, dim=0))
            f_z = sum(torch.chunk(f_z, chunks=2, dim=0)) / 2
            return f_z

        singular_values = compute_norms(encoder, x, x.shape[0]) ** 2

        fig, axs = plt.subplots()
        axs.plot(np.arange(len(singular_values)), singular_values)
        axs.set_xlabel("Feature dim")
        axs.set_ylabel("Singular Value")

        self.writer.add_figure('val/singular_values', fig, self.step)

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
    model = MyDistributedDataParallel(model, device_ids=[rank])
    _train_impl(rank, model, cfg)
