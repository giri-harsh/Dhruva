import numpy as np

from anchor.eval.calibration import IsotonicVarianceCalibrator, assess_calibration


def test_isotonic_fixes_a_shape_miscalibration_a_scalar_cannot():
    rng = np.random.default_rng(0)
    n = 16_000
    true_sigma = rng.uniform(0.4, 3.0, n)
    mu = rng.normal(0, 3, n)
    y = mu + rng.normal(0, true_sigma)
    # over-confident on small sigma, under-confident on large: no scalar T works
    pred_sigma = true_sigma * np.where(true_sigma < 1.2, 0.4, 2.0)
    lv = np.log(pred_sigma ** 2)

    h = n // 2
    iso = IsotonicVarianceCalibrator.fit(y[:h], mu[:h], lv[:h])
    before = assess_calibration(y[h:], mu[h:], lv[h:]).ece_sigma
    after = assess_calibration(y[h:], mu[h:], iso.apply(lv[h:])).ece_sigma
    assert before > 0.4
    assert after < 0.08


def test_isotonic_map_is_monotone_and_serialisable():
    rng = np.random.default_rng(1)
    n = 4000
    s = rng.uniform(0.5, 2.0, n)
    y = rng.normal(0, s); mu = np.zeros(n)
    iso = IsotonicVarianceCalibrator.fit(y, mu, np.log(s ** 2))
    assert np.all(np.diff(iso.y_knots) >= -1e-9)      # increasing
    j = iso.to_json()
    assert j["type"] == "isotonic_variance"
    assert len(j["sigma_pred_knots"]) == len(j["sigma_calibrated_knots"])
