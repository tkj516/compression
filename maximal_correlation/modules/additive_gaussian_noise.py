import math
import torch
import torch.nn as nn


class AdditiveGaussianNoise(nn.Module):
    def __init__(
        self,
        logvar: float = math.log(1 / 255.0 ** 2),
        train_noise_var: bool = False
    ):
        super().__init__()
        if train_noise_var:
            self.logvar = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            self.register_buffer("logvar", torch.tensor(logvar))

    def forward(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        samples = x + torch.exp(0.5 * self.logvar) * \
            torch.randn(num_samples, *x.shape, device=x.device)
        return samples
