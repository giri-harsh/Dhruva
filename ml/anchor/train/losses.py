"""Losses for ANCHOR-Net (PRD §6.6).

    L = L_speed + lambda_c * L_context + lambda_d * L_yaw

L_speed is a two-phase objective (probe: ml/docs/training-notes.md). Plain
Gaussian NLL trained jointly from scratch is degenerate — the variance head
learns fast, inflates sigma, and starves the mean head of gradient (measured:
NLL-from-scratch plateaus at val RMSE ~7.5 m/s and diverges; MSE-only reaches
~5.9). So:

  * epochs < warmup:  plain MSE on the mean (Head B rides along, not optimised)
  * epochs >= warmup: beta-NLL (Seitzer et al. 2022) —
        L = detach(eff_var ** beta) * NLL_per_sample
    with beta = 0.5. This down-weights the NLL's implicit 1/var on the mean
    gradient just enough to keep it healthy while the variance still calibrates.

`label_sigma` (per-sample, from ml/anchor/data/labels.py) is folded into the
effective variance so Head B learns the uncertainty ABOVE the known label
noise (training against pretend-perfect labels is what FR-08's calibration test
catches). A small L2 on pred_logvar guards against variance collapse.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOGVAR_L2 = 1e-3
LOGVAR_CLAMP = (-8.0, 6.0)
BETA_NLL = 0.5


def speed_loss(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    label_sigma: torch.Tensor | None = None,
    *,
    use_nll: bool,
    sample_weight: torch.Tensor | None = None,
    beta: float = BETA_NLL,
    logvar_l2: float = LOGVAR_L2,
) -> tuple[torch.Tensor, dict]:
    mu = pred_mean.reshape(-1)
    lv = pred_logvar.reshape(-1).clamp(*LOGVAR_CLAMP)
    y = target.reshape(-1)
    w = torch.ones_like(y) if sample_weight is None else sample_weight.reshape(-1)

    pred_var = torch.exp(lv)
    eff_var = pred_var if label_sigma is None else pred_var + label_sigma.reshape(-1) ** 2
    eff_var = eff_var.clamp_min(1e-6)

    se = (y - mu) ** 2
    if use_nll:
        nll = 0.5 * (torch.log(eff_var) + se / eff_var)
        weighting = eff_var.detach() ** beta                       # beta-NLL
        per_sample = weighting * nll
        reg = logvar_l2 * (lv ** 2).mean()
    else:
        per_sample = se                                            # MSE warm-up
        reg = 1e-4 * (lv ** 2).mean()                              # keep lv finite

    loss = (w * per_sample).sum() / w.sum().clamp_min(1.0) + reg
    with torch.no_grad():
        stats = {
            "nll": float((0.5 * (torch.log(eff_var) + se / eff_var)).mean()),
            "rmse": float(torch.sqrt((w * se).sum() / w.sum().clamp_min(1.0))),
            "bias": float((w * (mu - y)).sum() / w.sum().clamp_min(1.0)),
            "pred_sigma_mean": float(torch.sqrt(pred_var).mean()),
        }
    return loss, stats


# kept for the calibration eval path (no training, plain NLL is fine there)
def gaussian_nll(pred_mean, pred_logvar, target, label_sigma=None, *, logvar_l2=LOGVAR_L2):
    return speed_loss(pred_mean, pred_logvar, target, label_sigma,
                      use_nll=True, beta=0.0, logvar_l2=logvar_l2)


def context_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked cross-entropy — only the 3 CAN-derived classes have real labels
    (PRD §6.5). `mask` is 1 where the label is real."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[mask.bool()], labels[mask.bool()].long())
    return ce


def yaw_loss(pred_inc: torch.Tensor, pred_logvar: torch.Tensor, target_inc: torch.Tensor) -> torch.Tensor:
    loss, _ = gaussian_nll(pred_inc, pred_logvar, target_inc, None, logvar_l2=LOGVAR_L2)
    return loss
