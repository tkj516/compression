import torch
from torch import nn
from modules.layers import deconv5x5s2
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


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int = 3, hidden_channels: int = 192):
        super().__init__()

        modules = []
        hidden_dims = [512, 256, 128, 64, 32]

        self.inital = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_dims[0], kernel_size=1),
            nn.LeakyReLU(),
        )
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.body = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.body(self.inital(x)))
