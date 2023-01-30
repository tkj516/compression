import torch
from torch import nn
from modules.layers import conv5x5s2
from compressai.layers import GDN


class BMSHJEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 192):
        super().__init__()
        self.model = nn.Sequential(
            conv5x5s2(in_channels, hidden_channels),
            GDN(hidden_channels),
            conv5x5s2(hidden_channels, hidden_channels),
            GDN(hidden_channels),
            conv5x5s2(hidden_channels, hidden_channels),
            GDN(hidden_channels),
            conv5x5s2(hidden_channels, hidden_channels),          
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)