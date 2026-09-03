"""Losses for ANCHOR-Net (PRD §6.6).

    L = L_NLL + lambda_c * L_context + lambda_d * L_yaw

L_NLL is a heteroscedastic Gaussian negative log-likelihood on the mean-speed
target, using the model's own predicted variance (Head B). A per-sample label
uncertainty `label_sigma` (from ml/anchor/data/labels.py) is folded in so the
head learns the uncertainty that is ABOVE the known label noise — training
against pretend-perfect labels is what produces an overconfident variance head
(the failure FR-08's calibration test catches).

  effective_var = exp(pred_logvar) + label_sigma^2
  NLL = 0.5 * [ log(effective_var) + (y - mu)^2 / effective_var ]

A small L2 on pred_logvar (not on effective_var) keeps the head from collapsing
predicted variance to ~0 and leaning entirely on the label-noise floor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOGVAR_L2 = 1e-3          # lambda on ||pred_logvar||^2 (variance-collapse guard)
LOGVAR_CLAMP = (-8.0, 6.0)


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    label_sigma: torch.Tensor | None = None,
    *,
    logvar_l2: float = LOGVAR_L2,
) -> tuple[torch.Tensor, dict]:
    pred_mean = pred_mean.reshape(-1)
    pred_logvar = pred_logvar.reshape(-1).clamp(*LOGVAR_CLAMP)
    target = target.reshape(-1)

    pred_var = torch.exp(pred_logvar)
    if label_sigma is not None:
        eff_var = pred_var + label_sigma.reshape(-1) ** 2
    else:
        eff_var = pred_var
    eff_var = eff_var.clamp_min(1e-6)

    nll = 0.5 * (torch.log(eff_var) + (target - pred_mean) ** 2 / eff_var)
    loss = nll.mean() + logvar_l2 * (pred_logvar ** 2).mean()
    with torch.no_grad():
        stats = {
            "nll": float(nll.mean()),
            "rmse": float(torch.sqrt(F.mse_loss(pred_mean, target))),
            "pred_sigma_mean": float(torch.sqrt(pred_var).mean()),
        }
    return loss, stats


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
