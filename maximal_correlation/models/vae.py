import torch
from torch import nn


class GaussianVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()