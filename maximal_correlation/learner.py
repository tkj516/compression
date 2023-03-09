import os
import math
import dacite

import torch
import torchvision
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
from typing import List, Tuple

from models.svdc import SVDC
from losses.frobenius import NestedFrobeniusLoss, compute_norms
from ml_collections import ConfigDict
from dataset import ImageNetTrain, MNISTBase
from configs.base_configs import TrainerConfig
from compressai.optimizers import net_aux_optimizer
from modules.mnist_transform import MNISTTransform
from utils.class_builder import ClassBuilder


def normalize_l2ball(z, r):
    # normalize each row to have l2-norm <= r
    mask = (torch.norm(z, p=2, dim=-1) < r).float().unsqueeze(1)  # (B, 1)
    return mask * z + (1 - mask) * r * F.normalize(z, p=2, dim=1)


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
        

def nested_to_device(inputs: List[torch.Tensor], device: torch.device):
    for i, t in enumerate(inputs):
        inputs[i] = t.to(device)
    return inputs


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
        conf = {
            "net": {**{"type": cfg.optimizer_ed_config[0]}, **cfg.optimizer_ed_config[1]},
            "aux": {**{"type": cfg.optimizer_aux_config[0]}, **cfg.optimizer_aux_config[1]},
        }
        optimizer = net_aux_optimizer(self.model, conf)
        self.optimizer_ed, self.optimizer_aux = optimizer["net"], optimizer["aux"]
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_ed, "min")

        self.rd_lambda = cfg.rd_lambda
        self.nested_frobenius_loss = NestedFrobeniusLoss(
            end_indices=np.arange(1, self.model.encoder_config.out_channels + 1)
        )
        self.frobenius_lambda = cfg.frobenius_lambda

        # Insantiate a Tensorboard summary writer
        self.writer = SummaryWriter(self.trainer_config.model_dir)

    @property
    def is_master(self):
        return self.rank == 0

    def build_dataloaders(self):
        # self.dataset = ImageNetTrain()
        # TODO: The channel is now built into the dataloader
        self.dataset = MNISTBase(
            root="/fs/data/tejasj",
            download=True,
            transform=MNISTTransform(),  
        )
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
                inputs = nested_to_device(inputs, device)
                loss = self.train_step(inputs, logging_rank=self.rank == 0)

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

    def train_step(self, inputs: List[torch.Tensor], logging_rank: bool = False):
        self.optimizer_ed.zero_grad()
        self.optimizer_aux.zero_grad()

        # 1. Get two augmented inputs
        x, z, z_p = inputs

        # 2. Encode the augmented inputs
        f_zs = self.model.feature_encode(torch.cat([z, z_p], dim=0))
        f_z, f_z_p = torch.chunk(f_zs, chunks=2, dim=0)

        # 3. Compute the negative hscore loss between the encodings
        b, c, h, w = f_z.shape
        # reshaping for learning "patch" features
        phi = f_z.permute(0, 2, 3, 1).reshape(b * h * w, c)
        psi = f_z_p.permute(0, 2, 3, 1).reshape(b * h * w, c)
        # l2-ball projection
        phi = normalize_l2ball(phi, r=np.sqrt(self.cfg.mu))
        psi = normalize_l2ball(psi, r=np.sqrt(self.cfg.mu))
        # compute
        hscore_loss = self.nested_frobenius_loss(phi, psi) if self.step % self.trainer_config.hscore_freq == 0 else 0

        # 4. Mask the encoded augmented samples
        mask_percent = np.random.rand()
        mask = torch.ones(*f_z.shape)
        mask[..., : int(f_z.shape[-1] * (1 - mask_percent))] = 0.0
        f_z_m = f_z * mask.to(f_z.device)

        # 5. Encode and compress the image
        c = self.model.conditioner(f_z_m)
        w = self.model.encode(x, c)
        w_hat, w_likelihoods = self.model.entropy_bottleneck(w)

        # 6. Decode the image
        x_hat = self.model.decode(torch.cat([f_z_m, w_hat], dim=1))

        b, _, h, ww = x.shape
        num_pixels = b * h * ww
        bpp_loss = torch.log(w_likelihoods).sum() / \
            (-math.log(2) * num_pixels)
        mse_loss = F.mse_loss(x, x_hat)
        rd_loss = bpp_loss + self.rd_lambda * (255.0 ** 2) * mse_loss

        loss = rd_loss + self.frobenius_lambda * hscore_loss
        loss.backward()
        if self.trainer_config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm(
                self.model.parameters(), self.trainer_config.clip_max_norm)
        self.optimizer_ed.step()

        aux_loss = self.model.entropy_bottleneck.loss()
        aux_loss.backward()
        self.optimizer_aux.step()

        loss += aux_loss

        if logging_rank and self.step % self.trainer_config.log_every == 0:
            self.writer.add_scalar('train/mask_percent', mask_percent, self.step)
            self.writer.add_scalar('train/bpp', bpp_loss, self.step)
            self.writer.add_scalar('train/mse', mse_loss, self.step)
            self.writer.add_scalar('train/aux_loss', aux_loss, self.step)
            self.writer.add_scalar('train/rd_loss', rd_loss, self.step)
            self.writer.add_scalar('train/hscore_loss', hscore_loss, self.step)
            self.writer.add_scalar(
                'train/psnr', -10 / math.log(10) * torch.log(mse_loss), self.step)
            self.writer.add_images('train/gt', quantize_image(x), self.step)
            self.writer.add_images(
                'train/recon', quantize_image(x_hat), self.step)
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
        m = np.random.choice([0.25, 0.5, 0.75, 1.0])
        for inputs in tqdm(self.val_dataloader, desc=f"Running validation after step {self.step}"):
            inputs = nested_to_device(inputs, device)
            outputs = self.model(inputs, mask_percent=m)
            recon, likelihoods = outputs["x_hat"], outputs["w_likelihoods"]

            x = inputs[0]
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

        self.writer.add_scalar(f'val/{m}/bpp', bpp, self.step)
        self.writer.add_scalar(f'val/{m}/mse', mse, self.step)
        self.writer.add_scalar(f'val/{m}/aux_loss', aux, self.step)
        self.writer.add_scalar(f'val/{m}/rd_loss', loss, self.step)
        self.writer.add_scalar(f'val/{m}/psnr', -10 / math.log(10)
                               * torch.log(mse), self.step)
        self.writer.add_images(f'val/{m}/gt', quantize_image(x), self.step)
        self.writer.add_images(f'val/{m}/recon', quantize_image(recon), self.step)
        self.plot_singular_values(inputs)
        self.model.train()

        return loss

    def plot_singular_values(self, inputs: List[torch.Tensor]) -> None:
        def encoder(z: torch.Tensor) -> torch.Tensor:
            return self.model.feature_encode(z)

        _, z, _ = inputs
        singular_values = compute_norms(encoder, z, z.shape[0]) ** 2

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
