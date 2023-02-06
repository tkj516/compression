import torch

centralize = lambda phi: phi - torch.mean(phi, 0)  # centralize over batch dimension (when rows of phi are features)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NegativeHScore:  # Frobenius norm for maximal correlation (a.k.a. negative H-score)
    def __init__(self, feature_dim=0,
                 compute_over_batch=True, explicit_centering=True,
                 step=1, weights=None, stop_gradient=False):
        self.explicit_centering = explicit_centering
        self.compute_over_batch = compute_over_batch

        # parameters for nested objective
        self.feature_dim = feature_dim
        self.step = step
        self.weights = weights if weights else [1.] * feature_dim
        self.stop_gradient = stop_gradient

        self.end_indices = list(range(self.step, feature_dim + 1, self.step))
        if feature_dim not in self.end_indices:
            self.end_indices.append(feature_dim)
        print(f"The nested objective will structure the feature vector with the following end indices: "
              f"{self.end_indices} and weights {self.weights}")

    def __call__(self, phi, psi, buffer_psi):
        loss = 0
        prev_last_dim = 0

        # compute a nested objective
        for i in self.end_indices:
            if self.stop_gradient:
                partial_phi = torch.cat([phi[:, :prev_last_dim].detach(),
                                         phi[:, prev_last_dim:i]], dim=-1)
                partial_psi = torch.cat([psi[:, :prev_last_dim].detach(),
                                         psi[:, prev_last_dim:i]], dim=-1)
                partial_buffer_psi = torch.cat([buffer_psi[:, :prev_last_dim].detach(),
                                                buffer_psi[:, prev_last_dim:i]], dim=-1) if buffer_psi is not None else None
            else:
                partial_phi = phi[:, :i]
                partial_psi = psi[:, :i]
                partial_buffer_psi = buffer_psi[:, :i] if buffer_psi is not None else None
            loss += self.weights[min(i, self.feature_dim) - 1] * self._frobenius_norm(partial_phi, partial_psi, partial_buffer_psi)
            prev_last_dim = i

        return loss

    def _frobenius_norm(self, phi, psi, buffered_psi=None):
        # the reduction assumed here is `sum` (i.e., we take summation over batch)
        # phi, psi: (B, L)
        use_independent_batch = True
        if buffered_psi is None:
            use_independent_batch = False
            buffered_psi = psi
        if self.explicit_centering:
            phi = centralize(phi)
            psi = centralize(psi)

        batch_size = phi.shape[0]
        # note: unlike in FrobeniusNorm, we DO NOT normalize phi and psi by the batch size
        # loss1 (correlation) = -2 * E_{p(x,y)}[f^T(x) g(y)]
        loss1 = - 2 * (phi * psi).sum(-1).mean(0)  # scalar

        # compute loss2 = E_{p(x)p(y)}[(f^T(x) g(y))^2]
        if self.compute_over_batch:  # complexity = O(B^2 * L) + O(B^2)
            gram_matrix = phi @ buffered_psi.T  # (B, B); each entry is (f^T(x_i) g(y_j))
            if use_independent_batch:
                loss2 = (gram_matrix ** 2).mean()
            else:
                # if we compute this term using a single batch psi,
                # we should exclude the "paired" samples on the diagonal
                loss2 = (off_diagonal(gram_matrix) ** 2).mean()
        else:
            # compute tr(E_{p(x)}[f(x)f^T(x)] E_{p(y)}[g(y)g^T(y)])
            # complexity = O(B * L^2 + L^3) + O(L)
            lam_f = (phi.T @ phi) / batch_size  # (L, L)
            lam_g = (buffered_psi.T @ buffered_psi) / batch_size  # (L, L)
            loss2 = torch.trace(lam_f @ lam_g)

        loss = loss1 + loss2  # add a scalar to a scalar

        if not self.explicit_centering:
            loss += phi.mean(0) @ buffered_psi.mean(0)  # add a scalar to (B, )

        return loss * batch_size
