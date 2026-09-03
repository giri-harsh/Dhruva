import torch

from anchor.train.losses import speed_loss


def test_mse_phase_ignores_variance_head():
    mu = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.5, 2.0, 2.0])
    lv_a = torch.tensor([0.0, 0.0, 0.0])
    lv_b = torch.tensor([5.0, -3.0, 2.0])
    la, _ = speed_loss(mu, lv_a, y, None, use_nll=False)
    lb, _ = speed_loss(mu, lv_b, y, None, use_nll=False)
    # different logvars -> essentially same loss in MSE phase (only a tiny reg term)
    assert abs(float(la) - float(lb)) < 0.05


def test_nll_phase_rewards_correct_variance():
    mu = torch.zeros(2000)
    y = torch.randn(2000) * 2.0            # true sigma 2
    good_lv = torch.full((2000,), float(torch.log(torch.tensor(4.0))))   # var 4
    bad_lv = torch.full((2000,), float(torch.log(torch.tensor(0.04))))   # var 0.04
    lg, _ = speed_loss(mu, good_lv, y, None, use_nll=True, beta=0.0)
    lb, _ = speed_loss(mu, bad_lv, y, None, use_nll=True, beta=0.0)
    assert float(lg) < float(lb)


def test_sample_weight_downweights():
    mu = torch.tensor([0.0, 0.0])
    y = torch.tensor([0.0, 10.0])          # second sample has all the error
    lv = torch.zeros(2)
    full, _ = speed_loss(mu, lv, y, None, use_nll=False, sample_weight=torch.tensor([1.0, 1.0]))
    down, _ = speed_loss(mu, lv, y, None, use_nll=False, sample_weight=torch.tensor([1.0, 0.1]))
    assert float(down) < float(full)


def test_beta_nll_keeps_mean_gradient_alive_under_large_variance():
    """With beta=0.5 the mean gradient does not vanish when eff_var is large —
    the exact failure that made plain NLL degenerate."""
    mu = torch.zeros(64, requires_grad=True)
    y = torch.full((64,), 5.0)
    big_lv = torch.full((64,), 4.0)       # var ~55
    loss_beta, _ = speed_loss(mu, big_lv, y, None, use_nll=True, beta=0.5)
    loss_beta.backward()
    g_beta = mu.grad.abs().mean().item()

    mu2 = torch.zeros(64, requires_grad=True)
    loss_plain, _ = speed_loss(mu2, big_lv, y, None, use_nll=True, beta=0.0)
    loss_plain.backward()
    g_plain = mu2.grad.abs().mean().item()

    assert g_beta > 5 * g_plain            # beta-NLL restores mean gradient
