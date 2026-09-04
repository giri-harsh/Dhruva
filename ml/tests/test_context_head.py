import numpy as np
import pandas as pd
import torch

from anchor.data.context_labels import IDLE, NORMAL, ROUGH, ContextLabeller
from anchor.train.losses import context_loss


class _Seq:
    def __init__(self, df):
        self.df = df


def _mk(speed, jitter_amp):
    n = 40
    rng = np.random.default_rng(0)
    base = np.full(n, 30.0) + np.cumsum(rng.normal(0, jitter_amp, n))
    df = pd.DataFrame({
        "veh_speed_mps": np.full(n, speed),
        "veh_wheel_fl_radps": base, "veh_wheel_fr_radps": base + 0.01,
        "veh_wheel_rl_radps": base - 0.01, "veh_wheel_rr_radps": base,
    })
    return _Seq(df)


def test_context_labeller_idle_normal_rough():
    assert ContextLabeller(_mk(0.3, 0.0)).label(0, 20).cls == IDLE
    assert ContextLabeller(_mk(20.0, 0.01)).label(0, 20).cls == NORMAL
    assert ContextLabeller(_mk(20.0, 0.6)).label(0, 20).cls == ROUGH


def test_context_loss_ignores_classes_3_and_4():
    """logits for impulse/handling must not affect the loss (masked classes)."""
    torch.manual_seed(0)
    logits_a = torch.randn(64, 5)
    logits_b = logits_a.clone()
    logits_b[:, 3:] += 100.0                     # blow up the masked classes
    labels = torch.randint(0, 3, (64,))
    la = context_loss(logits_a, labels)
    lb = context_loss(logits_b, labels)
    assert torch.allclose(la, lb)


def test_context_loss_learns_the_three_real_classes():
    torch.manual_seed(0)
    x = torch.randn(512, 8)
    w = torch.randn(8, 3)
    labels = (x @ w).argmax(-1)
    head = torch.nn.Linear(8, 5)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = context_loss(head(x), labels)
        loss.backward(); opt.step()
    acc = (head(x)[:, :3].argmax(-1) == labels).float().mean().item()
    assert acc > 0.9


def test_model_context_head_shape():
    from anchor.models.anchornet import AnchorNet, AnchorNetConfig
    m = AnchorNet(AnchorNetConfig(enable_context_head=True))
    out = m(torch.randn(4, 20, 6))
    assert out["context_logits"].shape == (4, 5)
