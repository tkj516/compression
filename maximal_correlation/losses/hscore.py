import torch

centralize = lambda phi: phi - torch.mean(phi, 0)  # centralize over batch dimension (when rows of phi are features)
get_sample_correlation = lambda phi: (phi.T @ phi) / phi.shape[0]  # sample correlation (when rows of phi are features)


class NegativeHScore:  # Frobenius norm for maximal correlation
    def __init__(self, explicit_centering=True, compute_over_batch=True, nuclear_norm_weight=0.,
                 feature_dim=0, step=1, weights=None, stop_gradient=False):
        self.explicit_centering = explicit_centering
        self.compute_over_batch = compute_over_batch
        self.nuclear_norm_weight = nuclear_norm_weight

        # parameters for nested objective
        self.feature_dim = feature_dim
        self.step = step
        self.weights = weights if weights else [1.] * feature_dim
        self.stop_gradient = stop_gradient

    def __call__(self, phi, psi, buffer_psi):
        loss = 0
        prev_last_dim = 0

        # compute a nested objective
        for i in range(self.step, phi.shape[1], self.step):
            if self.stop_gradient:
                partial_phi = torch.concat(
                    [phi[:, :prev_last_dim].detach(),
                     phi[:, prev_last_dim:i]], dim=-1)
                partial_psi = torch.concat(
                    [psi[:, :prev_last_dim].detach(),
                     psi[:, prev_last_dim:i]], dim=-1)
                partial_buffer_psi = torch.concat(
                    [buffer_psi[:, :prev_last_dim].detach(),
                     buffer_psi[:, prev_last_dim:i]], dim=-1) if buffer_psi is not None else None
            else:
                partial_phi = phi[:, :i]
                partial_psi = psi[:, :i]
                partial_buffer_psi = buffer_psi[:, :i] if buffer_psi is not None else None
            loss += self.weights[i] * self.negative_hscore(partial_phi, partial_psi, partial_buffer_psi)
            prev_last_dim = i

        return loss

    def negative_hscore(self, phi, psi, buffered_psi=None):
        # the reduction assumed here is `sum` (i.e., we take summation over batch)
        # phi, psi: (B, L)
        if buffered_psi is None:
            buffered_psi = psi
        if self.explicit_centering:
            phi = centralize(phi)
            psi = centralize(psi)

        # correlation
        # NOTE: Added mean here
        loss = - 2 * (phi * psi).sum(-1).mean()  # (B, )

        # compute "correlation" term
        if self.compute_over_batch:  # complexity = B * B * L
            correlation = ((phi @ psi.T) ** 2).mean()
        else:  # complexity = B * L * L
            lam_f = get_sample_correlation(phi)  # (L, L)
            lam_g = get_sample_correlation(psi)  # (L, L)
            correlation = torch.trace(lam_f @ lam_g)
        loss += correlation  # add a scalar to (B, )

        if not self.explicit_centering:
            loss += phi.mean(0) @ buffered_psi.mean(0)  # add a scalar to (B, )

        # nuclear norm regularization based on its variational form
        regularizaton = 0.
        if self.nuclear_norm_weight > 0:
            regularizaton = .5 * ((phi ** 2).sum(-1) + (psi ** 2).sum(-1)).mean(0)  # (B, )

        return (loss + self.nuclear_norm_weight * regularizaton).sum(0)
