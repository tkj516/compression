import math
import os
import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
sys.path.append("..")

import dacite
import torch
from configs.base_configs import TrainerConfig
from dataset import MNISTBase
from learner import get_train_val_dataset, nested_to_device
from models.svdc import SVDC
from modules.mnist_transform import MNISTTransform
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
import torch.nn.functional as F


DEVICE = torch.device("cuda:0")


@torch.no_grad()
def compute_psnr_vs_mask_percent(
        checkpoint_dir: str,
        output_dir: str,
):
    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    cfg = ConfigDict(checkpoint["cfg"])
    trainer_config = dacite.from_dict(TrainerConfig, cfg.trainer_config)

    model = SVDC(**cfg.model_config).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = MNISTBase(
            root="/fs/data/tejasj",
            download=True,
            transform=MNISTTransform(),  
        )
    _, dataset = get_train_val_dataset(
        dataset, trainer_config.train_fraction)
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"Averaging of {len(dataset)} samples.")

    mses = []
    psnrs = []
    feature_length = model.feature_encoder_config.out_channels
    for mask_end_index in range(0, feature_length + 1):
        mse = 0
        psnr = 0
        for inputs in tqdm(dataloader):
            inputs = nested_to_device(inputs, DEVICE)
            outputs = model(inputs, mask_percent=1)
            
            x = inputs[0]
            recon = outputs["x_hat"]

            mse_batch = torch.mean(F.mse_loss(x, recon, reduction="none"), dim=[1, 2, 3])
            psnr_batch = -10 / math.log(10) * torch.log(mse_batch)

            mse += torch.sum(mse_batch)
            psnr += torch.sum(psnr_batch)
        mses.append(mse.cpu().numpy() / len(dataset))
        psnrs.append(psnr.cpu().numpy() / len(dataset))

    fig, ax = plt.subplots()
    ax.scatter(list(range(0, feature_length + 1)), psnrs)
    ax.set_ylabel("PSNR")
    ax.set_xlabel("Mask End Index")
    ax.set_title("PSNR vs. Mask End Index")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "psnr_vs_mask.png"))


if __name__ == "__main__":
    compute_psnr_vs_mask_percent(
        "/fs/data/tejasj/compression/maximal_correlation/checkpoints/svdc_mnist_v4/weights-375000.pt",
        "svdc_mnist_v4",
    )
