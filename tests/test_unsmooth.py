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


def test_rolling_theta_covers_both_ends():
    from src.altnow.unsmooth import rolling_theta
    y = pd.Series(np.random.default_rng(3).normal(size=60))
    th_c = rolling_theta(y, 20, 12, 0.0, 0.95, centered=True)
    th_t = rolling_theta(y, 20, 12, 0.0, 0.95, centered=False)
    assert th_c.notna().all()                       # 중심창: 양 끝 포함 전부 값이 있어야 한다
    assert np.isclose(th_c.iloc[-1], th_t.iloc[-1])  # 맨 끝은 후행창과 같다
    assert th_t.iloc[:11].isna().all() and th_t.iloc[11:].notna().all()
