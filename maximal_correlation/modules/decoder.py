import torch
from torch import nn
from layers import deconv5x5s2
from compressai.layers import GDN


class BMSHJDecoder(nn.Module):
    def __init__(self, out_channels: int = 3, hidden_channels: int = 192):
        super().__init__()
        self.model = nn.Sequential(
            deconv5x5s2(hidden_channels, hidden_channels),
            GDN(hidden_channels, inverse=True),
            deconv5x5s2(hidden_channels, hidden_channels),
            GDN(hidden_channels, inverse=True),
            deconv5x5s2(hidden_channels, hidden_channels),
            GDN(hidden_channels, inverse=True),
            deconv5x5s2(hidden_channels, out_channels),          
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)