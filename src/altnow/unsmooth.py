"""보고 수익률 언스무딩 — Geltner(1991) AR(p) 형.

    y_t  = Σ_{i=1..p} θ_i y_{t−i} + (1 − Σθ_i) r*_t
    r*_t = (y_t − Σ θ_i y_{t−i}) / (1 − Σθ_i)

- 평균은 보존된다 (E[r*] = E[y]). 변동성은 1/(1−Σθ) 근처로 커진다.
- θ 는 config 에 고정값이 없으면 전체 표본 AR(p) OLS(상수 포함)로 추정한다.
- 단순수익률에 그대로 적용한다 (엑셀 단위와 같다). 선형 변환이라 로그로 바꿔도 결과 차이는 작다.
- 차수·θ 는 config/nowcast.yaml `unsmoothing` 에서 바꾼다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg


def fit_theta(y: pd.Series, order: int) -> tuple[np.ndarray, np.ndarray]:
    """AR(order) OLS 계수와 표준오차 (상수 제외)."""
    if order == 0:
        return np.array([]), np.array([])
    f = AutoReg(y.dropna().values, lags=order, trend="c").fit()
    return np.asarray(f.params[1:]), np.asarray(f.bse[1:])


def unsmooth(y: pd.Series, theta: np.ndarray) -> pd.Series:
    """r*_t = (y_t − Σθ_i y_{t−i})/(1−Σθ). 앞 p 개 관측은 NaN."""
    y = y.dropna()
    p = len(theta)
    if p == 0:
        return y.copy()
    s = float(np.sum(theta))
    if s >= 1.0:
        raise ValueError(f"Σθ = {s:.3f} ≥ 1 — 언스무딩 불가")
    lagsum = sum(theta[i] * y.shift(i + 1) for i in range(p))
    return ((y - lagsum) / (1.0 - s)).dropna()


def series_config(cfg: dict, code: str) -> dict:
    u = cfg.get("unsmoothing") or {}
    s = (u.get("series") or {}).get(code) or {}
    return {"order": int(s.get("order", u.get("default_order", 0))), "theta": s.get("theta")}


def run_unsmoothing(cfg: dict, series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (언스무딩 수익률 code 별, 파라미터 표)."""
    out, rows = {}, []
    for code in series.columns:
        y = series[code].dropna()
        sc = series_config(cfg, code)
        p = sc["order"]
        if sc["theta"] is not None:
            th = np.atleast_1d(np.asarray(sc["theta"], dtype=float)); se = np.full(len(th), np.nan); src = "config"
            p = len(th)
        else:
            th, se = fit_theta(y, p); src = "estimated" if p else "-"
        r = unsmooth(y, th)
        out[code] = r
        lb = acorr_ljungbox((r - r.mean()).values, lags=[4], return_df=True)["lb_pvalue"].iloc[0] if len(r) > 8 else np.nan
        rows.append({"code": code, "order": p, "theta_source": src,
                     "theta": " / ".join(f"{t:.3f}" for t in th) if p else "",
                     "theta_se": " / ".join(f"{s:.3f}" for s in se) if p else "",
                     "sum_theta": float(np.sum(th)) if p else 0.0,
                     "mean_reported": float(y.mean()), "mean_unsmoothed": float(r.mean()),
                     "vol_reported_ann": float(y.std() * 2), "vol_unsmoothed_ann": float(r.std() * 2),
                     "vol_mult": float(r.std() / y.std()),
                     "ac1_reported": float(y.autocorr()), "ac1_unsmoothed": float(r.autocorr()),
                     "lb4_p_unsmoothed": float(lb), "n": int(len(r))})
    df = pd.DataFrame(out); df.index.name = "date"
    return df, pd.DataFrame(rows).set_index("code")


def to_labels(df_code: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for code, m in meta.iterrows():
        if code in df_code.columns:
            for lab in m["labels"]:
                out[lab] = df_code[code]
    return pd.DataFrame(out)
