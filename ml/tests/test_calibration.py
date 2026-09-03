import numpy as np

from anchor.eval.calibration import assess_calibration, reliability_diagram_points


def test_well_calibrated_model_has_low_ece_and_good_coverage():
    rng = np.random.default_rng(0)
    n = 20_000
    true_sigma = rng.uniform(0.2, 3.0, n)         # heteroscedastic
    mu = rng.normal(0, 5, n)
    y = mu + rng.normal(0, true_sigma)            # errors actually ~ N(0, true_sigma)
    pred_logvar = np.log(true_sigma ** 2)         # model knows the variance

    r = assess_calibration(y, mu, pred_logvar, n_bins=10)
    assert r.ece_sigma < 0.15
    emp1, exp1 = r.coverage["1.0"]
    assert abs(emp1 - exp1) < 0.03               # ~68% within 1 sigma
    assert r.pit_ks < 0.05
    pts = reliability_diagram_points(r)
    for sp, er in pts:                            # bins lie near y = x
        assert abs(sp - er) / er < 0.25


def test_overconfident_model_is_flagged():
    rng = np.random.default_rng(1)
    n = 20_000
    true_sigma = rng.uniform(0.5, 2.0, n)
    mu = rng.normal(0, 3, n)
    y = mu + rng.normal(0, true_sigma)
    pred_logvar = np.log((true_sigma * 0.3) ** 2)  # claims 3x more certain than it is

    r = assess_calibration(y, mu, pred_logvar, n_bins=10)
    assert r.ece_sigma > 0.4
    emp1, exp1 = r.coverage["1.0"]
    assert emp1 < exp1 - 0.1                       # far fewer than 68% land within 1 sigma
    assert r.pit_ks > 0.15                         # PIT far from uniform
