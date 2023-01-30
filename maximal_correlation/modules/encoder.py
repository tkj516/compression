import torch
from torch import nn
from modules.layers import conv5x5s2, ELICResidualBlock
from compressai.layers import GDN, AttentionBlock


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


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 192):
        super().__init__()
        
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        modules.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
        modules.append(nn.LeakyReLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ELICEncoder(nn.Module):
    """Implementation from paper.
    https://arxiv.org/pdf/2203.10886.pdf
    """

    def __init__(
        self, 
        in_channels: int = 3,
        hidden_channels: int = 192, 
        out_channels: int = 256,
        configuration: str = "L",
    ):
        super().__init__()
        if configuration == "S":
            self.model = nn.Sequential(
                conv5x5s2(in_channels, hidden_channels),
                ELICResidualBlock(hidden_channels, num_layers=1),
                conv5x5s2(hidden_channels, hidden_channels),
                ELICResidualBlock(hidden_channels, num_layers=1),
                conv5x5s2(hidden_channels, hidden_channels),
                ELICResidualBlock(hidden_channels, num_layers=1),
                conv5x5s2(hidden_channels, out_channels),
            )
        elif configuration == "L":
            self.model = nn.Sequential(
                conv5x5s2(in_channels, hidden_channels),
                ELICResidualBlock(hidden_channels, num_layers=3),
                conv5x5s2(hidden_channels, hidden_channels),
                ELICResidualBlock(hidden_channels, num_layers=3),
                AttentionBlock(hidden_channels),
                conv5x5s2(hidden_channels, hidden_channels),
                ELICResidualBlock(hidden_channels, num_layers=3),
                conv5x5s2(hidden_channels, out_channels),
                AttentionBlock(out_channels)
            )
        else:
            raise ValueError(f"Unknown configuration {configuration}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
