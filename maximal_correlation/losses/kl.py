import torch
from typing import Optional

class KLLoss:
    def __init__(
        self,
        compute_over_batch: bool = True,
        min_logvar: Optional[float] = None,
        max_logvar: Optional[float] = None,
    ):
        self.compute_over_batch = compute_over_batch
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def __call__(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.min_logvar and self.max_logvar:
            logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(
            torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2, 3])
        if self.compute_over_batch:
            kl = torch.mean(kl)
        return kl

if __name__ == "__main__":
    KLLoss()
