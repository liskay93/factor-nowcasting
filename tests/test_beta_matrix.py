"""베타 행렬·팩터 비중: λ=0 은 OLS, 블록 제약은 0, x = B'w 항등식, RC 합 = σ, macro_factor contrib 과 일치."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.altnow.beta_matrix import estimate_betas, factor_weights, weights_to_codes


def _data(seed=0, n=80):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2006-06-30", periods=n, freq="QE")
    fq = pd.DataFrame(rng.normal(0, 0.05, (n, 4)), index=idx, columns=["growth", "term", "inflation", "credit"])
    R = pd.DataFrame({"a": 0.6 * fq.growth + 0.2 * fq.credit + rng.normal(0, 0.02, n),
                      "b": 0.8 * fq.term + rng.normal(0, 0.01, n)}, index=idx)
    return R, fq


def test_ols_matches_statsmodels_and_block_zero():
    R, fq = _data()
    pri = {"default": ["growth", "term", "credit", "inflation"], "b": ["term", "credit", "growth", "inflation"]}
    B, D, info = estimate_betas(R, fq, {"method": "ols", "block": True, "n_block": 3}, pri)
    ols = sm.OLS(R.a, sm.add_constant(fq[["growth", "term", "credit"]])).fit()
    assert np.allclose(B.loc["a", ["growth", "term", "credit"]].values, ols.params[1:].values, atol=1e-10)
    assert B.loc["a", "inflation"] == 0.0 and B.loc["b", "inflation"] == 0.0
    assert abs(D["a"] - ols.resid.var(ddof=4) * 4) < 1e-12


def test_ridge_shrinks_toward_prior():
    R, fq = _data()
    pri = {"default": ["growth", "term", "credit", "inflation"]}
    B0, *_ = estimate_betas(R, fq, {"method": "ridge", "lambda": 1e6, "block": False}, pri)
    B1, *_ = estimate_betas(R, fq, {"method": "ridge", "lambda": 1e6, "block": False, "prior": {"a": {"growth": 0.7}}}, pri)
    assert abs(B0.loc["a", "growth"]) < 0.01          # 축소 목표 0
    assert abs(B1.loc["a", "growth"] - 0.7) < 0.01    # 축소 목표 사전값


def test_factor_weights_identities_and_macro_factor_match():
    R, fq = _data()
    pri = {"default": ["growth", "term", "credit", "inflation"]}
    B, D, _ = estimate_betas(R, fq, {"method": "ols", "block": False}, pri)
    w = pd.Series({"a": 0.6, "b": 0.4})
    res = factor_weights(B, D, w, fq)
    assert np.allclose(res["x"].values, (B.T @ w).values)
    assert abs(res["check"] - res["sigma_p"]) < 1e-12
    mf = Path("/home/user/macro_factor")
    if (mf / "src/risk/contrib.py").exists():        # macro_factor 의 오일러 분해와 같은 숫자여야 한다 (읽기 전용 import)
        import importlib.util
        spec = importlib.util.spec_from_file_location("mf_contrib", mf / "src/risk/contrib.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        ref = mod.euler_variance(B, w, fq.cov() * 4, D)
        assert abs(ref["sigma"] - res["sigma_p"]) < 1e-12
        assert np.allclose(ref["rc"].values, res["summary"]["RC"].values, atol=1e-12)


def test_weights_to_codes_merges_labels():
    meta = pd.DataFrame({"labels": [["A1", "A2"], ["B"]]}, index=["a", "b"])
    wl, wc = weights_to_codes({"A1": 0.3, "A2": 0.3, "B": 0.4}, meta, ["A1", "A2", "B"])
    assert abs(wc["a"] - 0.6) < 1e-12 and abs(wc["b"] - 0.4) < 1e-12
