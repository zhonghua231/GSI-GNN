
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

def square_contrastive_loss(
    logits: torch.Tensor,
    sims: Optional[torch.Tensor] = None,
    sim_weights: Optional[str] = 'identity',
    alpha: Optional[float] = 0.1
) -> torch.Tensor:

    if sims is not None:
        assert sims.device == logits.device
        assert sims.shape == logits.shape
        assert alpha is not None
        assert sim_weights is not None
        assert sim_weights in ('identity', 'exp', 'exp_thick_tail')

    N, device = logits.shape[0], logits.device

    if sims is None:
        row_targets = torch.arange(N, device=device)
        col_targets = row_targets
    else:

        if sim_weights == 'identity':
            weights = torch.ones((N,), device=device)
        elif sim_weights == 'exp':
            weights = (-1 * torch.arange(N, device=device)).exp()
        else:
            weights = (-1 * 1/(N ** 0.5) * torch.arange(N, device=device)).exp()

        weights = weights.unsqueeze(0).expand(N, -1)
        sort_inds = torch.argsort(sims, dim=1, descending=True)
        sims = (sims.gather(1, sort_inds) * weights)
        sims = sims.gather(1, sort_inds.argsort(1))

        row_reg_dist = sims - sims.min(dim=1).values.unsqueeze(1).expand(-1, N)
        row_reg_dist = F.normalize(row_reg_dist, p=1, dim=1)

        col_reg_dist = sims.T - sims.T.min(dim=1).values.unsqueeze(1).expand(-1, N)
        col_reg_dist = F.normalize(col_reg_dist, p=1, dim=1)

        row_targets = alpha * row_reg_dist + (1 - alpha) * torch.eye(N, device=device)
        col_targets = alpha * col_reg_dist + (1 - alpha) * torch.eye(N, device=device)

    return 0.5 * (
        F.cross_entropy(logits, row_targets) +
        F.cross_entropy(logits.T, col_targets)
    )
