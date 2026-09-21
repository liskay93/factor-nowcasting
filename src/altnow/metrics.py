"""OOS 평가 지표."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def hit_ratio(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.sign(y) == np.sign(yhat)))


def r2_oos(y: np.ndarray, yhat: np.ndarray, ybench: np.ndarray) -> float:
    """Campbell-Thompson R²_OOS: 1 − SSE(model)/SSE(benchmark)."""
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - ybench) ** 2))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM 통계량(HAC, lag h−1)과 양측 p 값. d = e1² − e2² (음수면 모델1 이 낫다)."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / n
    var = gamma0
    for k in range(1, h):
        gk = np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
        var += 2 * gk
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def summarize(y: pd.Series, preds: pd.DataFrame, bench_col: str = "mean", dm_vs: str = "ar1") -> pd.DataFrame:
    """preds: 예측 분기 × 모델. 각 모델의 RMSE·MAE·적중률·R²_OOS(대 bench_col)·DM(대 dm_vs)."""
    rows = {}
    yv = y.loc[preds.index].values
    for c in preds.columns:
        e = yv - preds[c].values
        row = {"n": len(yv), "rmse": rmse(e), "mae": mae(e), "hit": hit_ratio(yv, preds[c].values)}
        if bench_col in preds.columns:
            row[f"r2_oos_vs_{bench_col}"] = r2_oos(yv, preds[c].values, preds[bench_col].values)
        if dm_vs in preds.columns and c != dm_vs:
            row["dm_vs_" + dm_vs], row["dm_p"] = diebold_mariano(e, yv - preds[dm_vs].values)
        rows[c] = row
    return pd.DataFrame(rows).T
