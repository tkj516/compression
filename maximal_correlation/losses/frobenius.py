import numpy as np
import torch


# centralize over batch dimension (when rows of phi are features)
def centralize(phi): return phi - torch.mean(phi, 0)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# Frobenius norm for maximal correlation (a.k.a. negative H-score)
class NestedFrobeniusLoss:
    def __init__(
        self,
        end_indices=(),
        centering=True,
        stop_grad=False,
        verbose=False,
    ):
        self.centering = centering

        # parameters for nested objective
        self.stop_grad = stop_grad
        self.end_indices = list(end_indices)
        if verbose:
            print(f"The nested objective will structure the feature vector " 
                  f"with the following end indices {self.end_indices}")

    def __call__(self, phi, psi):
        loss = 0
        prev_last_dim = 0

        # compute a nested objective
        for i in self.end_indices:
            if self.stop_grad:
                partial_phi = torch.cat([phi[:, :prev_last_dim].detach(),
                                         phi[:, prev_last_dim:i]], dim=-1)
                partial_psi = torch.cat([psi[:, :prev_last_dim].detach(),
                                         psi[:, prev_last_dim:i]], dim=-1)
            else:
                partial_phi = phi[:, :i]
                partial_psi = psi[:, :i]
            loss += self._frobenius_norm(partial_phi, partial_psi)
            prev_last_dim = i

        return loss

    def _frobenius_norm(self, phi, psi):
        # the reduction assumed here is `mean` (i.e., we take mean over batch)
        # phi, psi: (B, L)
        # loss1 (correlation) = -2 * E_{p(x,y)}[f^T(x) g(y)]
        loss1 = - 2 * (phi * psi).sum(-1).mean(0)  # scalar

        # compute loss2 = E_{p(x)p(y)}[(f^T(x) g(y))^2]
        # unbiased version; fast if B << L
        gram_matrix = phi @ psi.T  # (B, B); each entry is (f^T(x_i) g(y_j))
        # since we compute this term using a single batch psi,
        # we should exclude the "paired" samples on the diagonal
        gram_matrix = off_diagonal(gram_matrix)
        loss2 = (gram_matrix ** 2).mean()  # scalar

        loss = loss1 + loss2  # add a scalar to a scalar

        if not self.centering:
            loss += 2 * gram_matrix.mean()  # add a scalar

        return loss


def compute_norms(model, xs, batch_size):
    # xs: (n, dim)
    n = len(xs)
    num_iters = (n // batch_size) + (n % batch_size != 0)
    coord_2norms = 0  # (feature_dim, )
    for i in range(num_iters):
        x = xs[i * batch_size:(i + 1) * batch_size]
        phi = model(x)  # (batch_size, feature_dim)
        if len(phi.shape) >= 2:
            b, c, h, w = phi.shape
            phi = phi.permute(0, 2, 3, 1).reshape(b * h * w, c)
        coord_2norms += (phi ** 2).sum(0).data.cpu().numpy()
    coord_2norms = np.sqrt(coord_2norms / n)

    return coord_2norms  # (feature_dim, )
