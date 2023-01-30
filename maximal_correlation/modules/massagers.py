import torch
import torch.nn as nn

class SimpleMassager(nn.Module):
    def __init__(self, channels: int = 192):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NoMassager(nn.Module):
    def __init__(self, **kwargs):
        del kwargs
        super().__init__()
        self.model = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)