"""ANCHOR-Net — one dilated temporal-convolution trunk, multiple heads.

PRD §6.5:
  * dilated TCN trunk, kernel 3, dilations 1/2/4, receptive field 15 timesteps
    (1.5 s, inside the 2.0 s / 20-sample window),
  * parameter target < 50 000, int8-quantisable,
  * Head A (mean): softplus, >= 0,
  * Head B (log-variance): linear — trained with Gaussian NLL + a small L2 on the
    log-variance to stop variance collapse. This is the primary novelty claim
    (FR-08), measured by the calibration test, not asserted.
  * Head C (motion context, S-15 Should) and Head D (yaw increment, S-16 Should)
    are built but off by default — enabled per config, gated on their ablations.

Rejected alternatives are kept as switchable trunks for the ablation table
(PRD §6.5): `trunk="gru"` builds the recurrent variant (ablation row 12).

Contract coupling: the exported graph's I/O is frozen in
`contracts/model_io/generate_stub_model.py`. This module's forward() returns a
dict keyed by the contract output names so the exporter is a thin wrapper.
Input is [B, T=20, F=6] AFTER normalisation (the caller applies
`(x-mean)/std`; normalisation is NOT in the graph — contracts/model_io §2.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import (
    NUM_FEATURES,
    OUTPUT_LOGVAR_NAME,
    OUTPUT_MEAN_NAME,
    WINDOW_SIZE_SAMPLES,
)

CTX_CLASSES = ("idle", "normal", "rough", "impulse", "handling")


@dataclass
class AnchorNetConfig:
    trunk: str = "tcn"                 # "tcn" | "gru"
    hidden: int = 40
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    tcn_kernel: int = 3
    dropout: float = 0.05
    enable_context_head: bool = False  # Head C (S-15)
    enable_yaw_head: bool = False      # Head D (S-16 / FR-32)
    context_classes: tuple[str, ...] = field(default=CTX_CLASSES)


class _ChannelNorm(nn.Module):
    """Per-timestep normalisation over the channel axis (like LayerNorm(C) at
    each t), written with primitive ops only — NOT nn.LayerNorm — so the
    exported graph never needs the opset-17 `LayerNormalization` op that
    TOOLCHAIN.md flags as an ONNX-Runtime-Mobile risk. Also keeps the trunk's
    receptive field genuinely local (GroupNorm/global LayerNorm would pool over
    time and make every output depend on every input)."""
    def __init__(self, ch: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x):                    # x: [B, C, T], normalise over C
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class _DilatedBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm = _ChannelNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                    # x: [B, C, T]
        y = self.conv(x)                     # symmetric padding preserves T
        y = self.drop(F.relu(self.norm(y)))
        return x + y                         # residual


class _TCNTrunk(nn.Module):
    def __init__(self, cfg: AnchorNetConfig):
        super().__init__()
        self.inproj = nn.Conv1d(NUM_FEATURES, cfg.hidden, 1)
        self.blocks = nn.ModuleList(
            _DilatedBlock(cfg.hidden, cfg.tcn_kernel, d, cfg.dropout)
            for d in cfg.tcn_dilations
        )

    def forward(self, x):                    # x: [B, T, F]
        h = self.inproj(x.transpose(1, 2))   # [B, H, T]
        for blk in self.blocks:
            h = blk(h)
        return h.mean(dim=-1)                 # global average pool -> [B, H]


class _GRUTrunk(nn.Module):
    """Ablation row 12. A recurrent hidden state carried across an outage is a
    hidden integrator — the exact accumulating-error structure the thesis
    removes — so this is reported, not shipped, unless it wins (PRD §6.5)."""
    def __init__(self, cfg: AnchorNetConfig):
        super().__init__()
        self.gru = nn.GRU(NUM_FEATURES, cfg.hidden, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class _Head(nn.Module):
    def __init__(self, hidden: int, out: int, final: str | None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out),
        )
        self.final = final

    def forward(self, z):
        y = self.net(z)
        if self.final == "softplus":
            y = F.softplus(y)
        return y


class AnchorNet(nn.Module):
    def __init__(self, cfg: AnchorNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or AnchorNetConfig()
        self.trunk = _TCNTrunk(self.cfg) if self.cfg.trunk == "tcn" else _GRUTrunk(self.cfg)
        h = self.cfg.hidden
        self.head_mean = _Head(h, 1, final="softplus")     # Head A: mean speed >= 0
        self.head_logvar = _Head(h, 1, final=None)         # Head B: log-variance
        self.head_context = (_Head(h, len(self.cfg.context_classes), None)
                             if self.cfg.enable_context_head else None)
        self.head_yaw = (_Head(h, 2, None)                 # (yaw_increment, yaw_logvar)
                         if self.cfg.enable_yaw_head else None)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[1:] != (WINDOW_SIZE_SAMPLES, NUM_FEATURES):
            raise ValueError(f"expected [B, {WINDOW_SIZE_SAMPLES}, {NUM_FEATURES}], got {tuple(x.shape)}")
        z = self.trunk(x)
        out = {
            OUTPUT_MEAN_NAME: self.head_mean(z),
            OUTPUT_LOGVAR_NAME: self.head_logvar(z),
        }
        if self.head_context is not None:
            out["context_logits"] = self.head_context(z)
        if self.head_yaw is not None:
            yaw = self.head_yaw(z)
            out["yaw_increment_rad"] = yaw[:, :1]
            out["yaw_log_variance"] = yaw[:, 1:]
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
