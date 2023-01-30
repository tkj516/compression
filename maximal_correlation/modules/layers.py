import torch
from torch import nn
import functools

def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int, 
    stride: int,
) -> nn.Module:
    return nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=kernel_size//2
    )

conv1x1s1 = functools.partial(conv, kernel_size=1, stride=1)
conv3x3s1 = functools.partial(conv, kernel_size=3, stride=1)
conv3x3s2 = functools.partial(conv, kernel_size=3, stride=2)
conv5x5s1 = functools.partial(conv, kernel_size=5, stride=1)
conv5x5s2 = functools.partial(conv, kernel_size=5, stride=2)


def deconv(
    in_channels: int, 
    out_channels: int, 
    kernel_size: int, 
    stride: int,
):
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

deconv3x3s1 = functools.partial(deconv, kernel_size=3, stride=1)
deconv3x3s2 = functools.partial(deconv, kernel_size=3, stride=2)
deconv5x5s1 = functools.partial(deconv, kernel_size=5, stride=1)
deconv5x5s2 = functools.partial(deconv, kernel_size=5, stride=2)


class ELICResidualUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.model = nn.Sequential(
            conv1x1s1(channels, channels // 2),
            nn.ReLU(),
            conv3x3s1(channels // 2, channels // 2),
            nn.ReLU(),
            conv1x1s1(channels // 2, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class ELICResidualBlock(nn.Module):
    def __init__(self, channels: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ELICResidualUnit(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
        