import torch
import torch.nn as nn


class AdditiveGaussianNoise(nn.Module):
    def __init__(
        self,
        logvar: float = 0.0,
        num_avg_samples: int = 16,
        train_noise_var: bool = False
    ):
        super().__init__()
        self.num_avg_samples = num_avg_samples

        if train_noise_var:
            self.logvar = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            self.register_buffer("logvar", torch.tensor(logvar))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = x + torch.exp(self.logvar) * \
            torch.randn(self.num_avg_samples, *x.shape, device=x.device())
        return samples.mean(0)
