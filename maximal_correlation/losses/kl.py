import torch

class KLLoss:
    def __init__(
        self,
        compute_over_batch: bool = True,
        min_logvar: float = -30.0,
        max_logvar: float = 20.0,
    ):
        self.compute_over_batch = compute_over_batch
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def __call__(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(
            torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2, 3])
        if self.compute_over_batch:
            kl = torch.mean(kl)
        return kl

if __name__ == "__main__":
    KLLoss()
