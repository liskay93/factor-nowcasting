"""언스무딩: 차수 0 은 항등, 차수 1 은 Geltner 식, 평균 보존, 스무딩된 합성 자료에서 θ·원 수익률 복원."""
import numpy as np
import pandas as pd

from src.altnow.unsmooth import fit_theta, unsmooth


def test_order0_identity():
    y = pd.Series(np.random.default_rng(0).normal(size=50))
    assert unsmooth(y, np.array([])).equals(y)


def test_geltner_formula_and_mean():
    y = pd.Series(np.random.default_rng(1).normal(0.01, 0.03, 200))
    th = np.array([0.4])
    r = unsmooth(y, th)
    manual = (y.iloc[1:] - 0.4 * y.shift(1).iloc[1:]) / 0.6
    assert np.allclose(r.values, manual.values)
    assert abs(r.mean() - y.mean()) < 5e-3


def test_recovers_true_returns():
    rng = np.random.default_rng(2)
    true = rng.normal(0.02, 0.05, 2000)
    theta = 0.5
    y = np.zeros_like(true)
    for t in range(1, len(true)):
        y[t] = theta * y[t - 1] + (1 - theta) * true[t]
    ys = pd.Series(y[1:])
    th, _ = fit_theta(ys, 1)
    assert abs(th[0] - theta) < 0.05
    r = unsmooth(ys, np.array([theta]))
    assert np.allclose(r.values, true[2:], atol=1e-10)
