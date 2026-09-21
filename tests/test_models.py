"""모델: ridge λ=0 은 OLS, 상태공간은 항등식(누적기·Geltner 관측식)을 만족해야 한다."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.altnow.models.ridge_dl import RidgeDL, _ridge_fit
from src.altnow.models.statespace import fit_ss, monthly_frame


def _synthetic(n_m=1200, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2005-01-31", periods=n_m, freq="ME")
    F = pd.DataFrame(rng.normal(0, 0.03, (n_m, 2)), index=months, columns=["g", "t"])
    alpha, beta, sig, theta = 0.004, np.array([0.5, -0.2]), 0.01, 0.4
    rstar = alpha + F.values @ beta + rng.normal(0, sig, n_m)
    y, prev = {}, 0.0
    for i in range(2, n_m, 3):
        q = months[i]
        true_q = rstar[i - 2:i + 1].sum()
        y[q] = theta * prev + (1 - theta) * true_q
        prev = y[q]
    return F, pd.Series(y), (alpha, beta, sig, theta)


def test_ridge_zero_equals_ols():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    y = pd.Series(X.values @ np.array([1, -2, 0.5, 0, 3]) + rng.normal(size=60))
    m = RidgeDL(alphas=[0.0], cv_min_train=20).fit(X, y)
    ols = sm.OLS(y.values, sm.add_constant(X.values)).fit()
    assert np.allclose(m.coef_unscaled().values, ols.params[1:], atol=1e-10)
    assert np.allclose(m.predict(X), ols.fittedvalues, atol=1e-10)


def test_ridge_shrinks():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(60, 5)))
    y = pd.Series(rng.normal(size=60))
    c0 = RidgeDL(alphas=[0.0], cv_min_train=20).fit(X, y).coef_.abs().sum()
    c1 = RidgeDL(alphas=[100.0], cv_min_train=20).fit(X, y).coef_.abs().sum()
    assert c1 < c0


def test_statespace_recovers_parameters_and_identities():
    F, y, (alpha, beta, sig, theta) = _synthetic()
    endog, Fm, ylag = monthly_frame(F, y, fill_mean=F.mean())
    r = fit_ss(endog, Fm, ylag)
    assert r.res.mle_retvals["converged"]
    assert abs(r.theta - theta) < 0.06
    assert np.allclose(r.beta, beta, atol=0.05)
    assert abs(r.alpha - alpha) < 0.003
    # 항등식: 관측 분기 y_q = θ y_{q-1} + (1-θ) C_q  (관측오차 0)
    C = pd.Series(r.res.smoothed_state[0], index=r.months)
    u = r.unsmoothed_monthly()
    for q in endog.dropna().index[-5:]:
        assert np.isclose(endog[q], r.theta * ylag[q] + (1 - r.theta) * C[q], atol=1e-8)
        assert np.isclose(C[q], u.loc[q - pd.offsets.MonthEnd(2): q].sum(), atol=1e-10)
    # 관측이 없는 분기의 nowcast = θ y_{q-1} + (1-θ)(3α + β'ΣF)
    q = endog.dropna().index[-1]
    mu, _ = r.nowcast(q)
    assert np.isfinite(mu)
