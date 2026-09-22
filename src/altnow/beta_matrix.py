"""베타 행렬 B (시리즈 × 팩터) 와 팩터 비중.

    x   = B'w                      팩터 비중 (팩터 포트폴리오의 명목 비중; growth 0.35 = ACWI 35% 와 같은 노출)
    x·σ                            1σ 충격 손익 (팩터 간 크기 비교용)
    RC_k = x_k (Ωx)_k / σ_p        오일러 위험기여 (Σ = BΩB' + D). macro_factor src/risk/contrib.py 와 같은 식

추정: OLS 또는 ridge. ridge 는 표준화 팩터 위에서, λ 는 공통(전 시리즈 CV MSE 합 최소)·시리즈별·고정.
블록 제약: beta_priority 의 앞 n_block 개 팩터만 쓰고 나머지 β = 0.
사전값(prior): 축소 목표를 0 이 아니라 β₀ 로 둔다 → y − Xβ₀ 를 ridge 하고 β = β₀ + β̂.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


def _prep(R: pd.DataFrame, fq: pd.DataFrame, code: str, allowed: list[str]) -> pd.DataFrame | None:
    d = pd.concat([R[code].rename("y"), fq[allowed]], axis=1).dropna()
    return d if len(d) else None


def _fit_one(X: np.ndarray, y: np.ndarray, lam: float, standardize: bool, b0: np.ndarray) -> tuple[np.ndarray, float, float]:
    """반환 (β 원단위, R², 잔차분산). lam=0 → OLS."""
    y_adj = y - X @ b0
    mu, sd = X.mean(0), (X.std(0, ddof=0) if standardize else np.ones(X.shape[1]))
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    m = Ridge(alpha=lam, fit_intercept=True).fit(Xs, y_adj)
    b = b0 + m.coef_ / sd
    resid = y - (m.intercept_ - (mu / sd) @ m.coef_ + X @ b)
    r2 = 1 - resid.var() / y.var()
    return b, float(r2), float(resid.var(ddof=X.shape[1] + 1))


def _cv_mse(X: np.ndarray, y: np.ndarray, lam: float, standardize: bool, b0: np.ndarray, splits: int) -> float:
    mse = 0.0
    for tr, te in TimeSeriesSplit(n_splits=splits).split(X):
        b, _, _ = _fit_one(X[tr], y[tr], lam, standardize, b0)
        c = y[tr].mean() - X[tr].mean(0) @ b
        mse += float(((y[te] - (c + X[te] @ b)) ** 2).mean())
    return mse / splits


def estimate_betas(R: pd.DataFrame, fq: pd.DataFrame, bm: dict, priority: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """반환: B (시리즈 × 팩터, 허용 안 된 팩터 0), D (잔차분산 연율), info (λ, R², n, 허용 팩터)."""
    F = list(fq.columns)
    method = bm.get("method", "ols")
    standardize = bool(bm.get("standardize", True))
    alphas = [float(a) for a in bm.get("alphas", [0.01, 0.1, 1, 10, 100])]
    splits = int(bm.get("cv_splits", 5))
    n_block = int(bm.get("n_block", len(F)))
    priors = bm.get("prior") or {}
    data, allowed_map, b0_map = {}, {}, {}
    for c in R.columns:
        allowed = (priority.get(c, priority.get("default", F))[:n_block] if bm.get("block", True) else F)
        d = _prep(R, fq, c, allowed)
        if d is None or len(d) < int(bm.get("min_obs", 24)):
            continue
        data[c] = d; allowed_map[c] = allowed
        b0_map[c] = np.array([float((priors.get(c) or {}).get(f, 0.0)) for f in allowed])
    # λ 결정
    lam_cfg = bm.get("lambda", "common")
    lam_map = {}
    if method == "ols":
        lam_map = {c: 0.0 for c in data}
    elif isinstance(lam_cfg, (int, float)):
        lam_map = {c: float(lam_cfg) for c in data}
    elif lam_cfg == "per_series":
        for c, d in data.items():
            X, y = d[allowed_map[c]].values, d["y"].values
            lam_map[c] = min(alphas, key=lambda a: _cv_mse(X, y, a, standardize, b0_map[c], splits))
    else:  # common
        pooled = {a: 0.0 for a in alphas}
        for c, d in data.items():
            X, y = d[allowed_map[c]].values, d["y"].values
            for a in alphas:
                pooled[a] += _cv_mse(X, y, a, standardize, b0_map[c], splits) / y.var()
        lam = min(pooled, key=pooled.get)
        lam_map = {c: lam for c in data}
    rows, B, D = [], {}, {}
    for c, d in data.items():
        X, y = d[allowed_map[c]].values, d["y"].values
        b, r2, rv = _fit_one(X, y, lam_map[c], standardize, b0_map[c])
        B[c] = pd.Series(0.0, index=F); B[c][allowed_map[c]] = b
        D[c] = rv * 4.0
        rows.append({"code": c, "lambda": lam_map[c], "r2": r2, "n": len(d), "allowed": ",".join(allowed_map[c]),
                     "resid_vol_ann": float(np.sqrt(rv * 4)), "start": d.index[0].date(), "end": d.index[-1].date()})
    return pd.DataFrame(B).T[F], pd.Series(D), pd.DataFrame(rows).set_index("code")


def weights_to_codes(weights: dict | str, meta: pd.DataFrame, labels: list[str]) -> tuple[pd.Series, pd.Series]:
    """라벨 비중 → 코드 비중 (같은 원지수 라벨은 합산). 'equal' 이면 라벨 균등."""
    if weights == "equal":
        wl = pd.Series(1.0 / len(labels), index=labels)
    else:
        wl = pd.Series({k: float(v) for k, v in weights.items()})
        bad = [l for l in wl.index if l not in labels]
        if bad:
            raise KeyError(f"엑셀 라벨에 없는 이름: {bad}")
    wl = wl / wl.sum()
    wc = pd.Series(0.0, index=meta.index)
    for code, m in meta.iterrows():
        wc[code] = sum(wl.get(l, 0.0) for l in m["labels"])
    return wl, wc[wc > 0]


def factor_weights(B: pd.DataFrame, D: pd.Series, w: pd.Series, fq: pd.DataFrame) -> dict:
    """x = B'w, 1σ 손익, 오일러 RC. Ω 는 분기 팩터 공분산 연율화."""
    keep = [c for c in w.index if c in B.index]
    wk = w[keep] / w[keep].sum()
    Bk = B.loc[keep]
    contrib = Bk.mul(wk, axis=0)                     # 자산별 w_i β_i
    x = contrib.sum()
    sig = fq[B.columns].std() * 2
    Om = fq[B.columns].cov() * 4
    Ox = Om @ x
    var_f = float(x @ Ox); var_e = float((wk ** 2 * D.reindex(keep).fillna(0.0)).sum())
    s = float(np.sqrt(var_f + var_e))
    rc = x * Ox / s
    summary = pd.DataFrame({"factor_weight_x": x, "sigma_F": sig, "pnl_1sigma": x * sig, "RC": rc, "RC_pct": rc / s * 100})
    return {"w_code": wk, "contrib": contrib, "x": x, "summary": summary, "sigma_p": s,
            "rc_resid": var_e / s, "rc_resid_pct": var_e / s / s * 100, "check": float(rc.sum() + var_e / s)}


def rolling_factor_weights(R: pd.DataFrame, fq: pd.DataFrame, bm: dict, priority: dict, w: pd.Series,
                           window: int = 20, lam: float | None = None) -> pd.DataFrame:
    """롤링 창마다 B 를 다시 추정해 x_t = B_t'w. λ 는 전체표본에서 고른 값을 고정해 쓴다."""
    F = list(fq.columns)
    bm2 = dict(bm); bm2["lambda"] = lam if lam is not None else 0.0
    bm2["min_obs"] = min(int(bm.get("min_obs", 24)), window)
    idx = fq.index
    rows = {}
    for i in range(window - 1, len(idx)):
        end = idx[i]; start = idx[i - window + 1]
        Rw = R.loc[start:end]; fw = fq.loc[start:end]
        if Rw[w.index].dropna().shape[0] < bm2["min_obs"]:
            continue
        try:
            B, D, _ = estimate_betas(Rw[w.index], fw, bm2, priority)
        except Exception:
            continue
        if B.empty:
            continue
        keep = [c for c in w.index if c in B.index]
        if len(keep) < len(w) * 0.6:
            continue
        wk = w[keep] / w[keep].sum()
        rows[end] = B.loc[keep].mul(wk, axis=0).sum()
    out = pd.DataFrame(rows).T
    out.index.name = "date"
    return out
