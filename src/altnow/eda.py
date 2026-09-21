"""탐색적 분석 — 보고(smoothed) vs 언스무딩(unsmoothed) 두 기준을 같은 표·그림에 나란히 둔다.

레짐은 macro_factor 의 월간 라벨(data/processed/regimes_monthly.csv, scripts/export_regimes.py) 을 분기로 모은다:
분기 안 세 달의 다수 라벨, 동률이면 마지막 달.
연율화: 평균 ×4 (산술), 변동성 ×2. 샤프 = (평균 − 3M T-bill 분기율) / 표준편차 × 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASES = {"reported": "보고(스무딩)", "unsmoothed": "언스무딩"}
REGIME_EN = {  # 그림용 (한글 폰트 없음)
    "확장": "Expansion", "침체": "Recession",
    "성장↑인플레↑": "G+ I+", "성장↑인플레↓": "G+ I-", "성장↓인플레↑": "G- I+", "성장↓인플레↓": "G- I-",
    "인상": "Hike", "인하": "Cut", "동결": "Hold", "리스크온": "Risk-on", "리스크오프": "Risk-off",
    "확장기": "03-07 Expansion", "금융위기": "08-09 GFC", "양적완화·유럽위기·테이퍼": "10-13 QE/Euro/Taper",
    "디스인플레·유가급락": "14-16 Disinfl/Oil", "후기 사이클": "17-19 Late cycle", "코로나": "2020 COVID",
    "인플레이션·긴축": "21-22 Infl/Tightening", "디스인플레·고금리": "23- Disinfl/High rates",
}
FAMILY_EN = {"R1_경기": "R1 business cycle (USREC)", "R2_성장인플레": "R2 growth x inflation (indpro_yoy | bei10)",
             "R3_통화정책": "R3 monetary policy (FEDFUNDS 3m chg)", "R4_위험선호": "R4 risk appetite (VIX top 20%)", "R5_서사": "R5 narrative"}
REGIME_ORDER = {
    "R1_경기": ["확장", "침체"],
    "R2_성장인플레": ["성장↑인플레↑", "성장↑인플레↓", "성장↓인플레↑", "성장↓인플레↓"],
    "R3_통화정책": ["인상", "동결", "인하"],
    "R4_위험선호": ["리스크온", "리스크오프"],
    "R5_서사": ["확장기", "금융위기", "양적완화·유럽위기·테이퍼", "디스인플레·유가급락", "후기 사이클", "코로나", "인플레이션·긴축", "디스인플레·고금리"],
}


def regimes_quarterly(rm: pd.DataFrame) -> pd.DataFrame:
    """월간 라벨 → 분기 라벨 (다수결, 동률은 마지막 달)."""
    out = {}
    for col in rm.columns:
        s = rm[col].dropna()
        g = s.groupby(s.index.to_period("Q"))
        def pick(x):
            vc = x.value_counts()
            top = vc[vc == vc.max()].index
            return x.iloc[-1] if len(top) > 1 else top[0]
        q = g.apply(pick)
        q.index = q.index.to_timestamp(how="end").normalize()
        out[col] = q
    df = pd.DataFrame(out)
    df.index.name = "date"
    return df


def cash_quarterly(dgs3mo_daily: pd.Series) -> pd.Series:
    """DGS3MO(%) 일간 → 분기 평균 /100 /4 = 분기 무위험 수익률."""
    q = dgs3mo_daily.resample("QE").mean() / 100.0 / 4.0
    q.index = q.index + pd.offsets.QuarterEnd(0)
    return q


def max_drawdown(r: pd.Series) -> float:
    lv = (1 + r).cumprod()
    return float((lv / lv.cummax() - 1).min())


def summary_stats(R: dict[str, pd.DataFrame], rf: pd.Series) -> pd.DataFrame:
    rows = []
    for base, df in R.items():
        for c in df.columns:
            r = df[c].dropna()
            ex = r - rf.reindex(r.index).fillna(rf.mean())
            rows.append({"base": base, "code": c, "n": len(r), "start": r.index[0].date(), "end": r.index[-1].date(),
                         "ann_return_geo": float((1 + r).prod() ** (4 / len(r)) - 1), "ann_mean": float(r.mean() * 4),
                         "ann_vol": float(r.std() * 2), "sharpe": float(ex.mean() / r.std() * 2),
                         "skew": float(r.skew()), "kurt": float(r.kurt()), "worst_q": float(r.min()), "best_q": float(r.max()),
                         "max_dd": max_drawdown(r), "ac1": float(r.autocorr()), "hit": float((r > 0).mean())})
    return pd.DataFrame(rows).set_index(["base", "code"])


def regime_stats(R: dict[str, pd.DataFrame], rq: pd.DataFrame, family: str, min_n: int = 4) -> pd.DataFrame:
    """시리즈 × 상태 × 기준: 연율 평균·변동성·n·적중률·최악 분기."""
    lab = rq[family].dropna()
    rows = []
    for base, df in R.items():
        for c in df.columns:
            r = df[c].dropna()
            common = r.index.intersection(lab.index)
            for state in REGIME_ORDER[family]:
                idx = common[lab.loc[common] == state]
                x = r.loc[idx]
                if len(x) < min_n:
                    rows.append({"base": base, "code": c, "state": state, "n": len(x)})
                    continue
                rows.append({"base": base, "code": c, "state": state, "n": len(x), "ann_mean": float(x.mean() * 4),
                             "ann_vol": float(x.std() * 2), "hit": float((x > 0).mean()), "worst_q": float(x.min()),
                             "t_mean": float(x.mean() / (x.std() / np.sqrt(len(x))))})
    return pd.DataFrame(rows).set_index(["base", "code", "state"])


def factor_betas(R: dict[str, pd.DataFrame], fq: pd.DataFrame, rq: pd.DataFrame | None = None,
                 cond_family: str | None = None) -> pd.DataFrame:
    """분기 팩터 4개에 대한 OLS 베타 (HAC t). cond_family 를 주면 상태별로 따로 추정한다."""
    rows = []
    for base, df in R.items():
        for c in df.columns:
            d = pd.concat([df[c].rename("y"), fq], axis=1).dropna()
            states = [None] if cond_family is None else REGIME_ORDER[cond_family]
            for st in states:
                dd = d if st is None else d.loc[d.index.intersection(rq.index[rq[cond_family] == st])]
                if len(dd) < 12:
                    continue
                X = sm.add_constant(dd[fq.columns])
                f = sm.OLS(dd["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
                row = {"base": base, "code": c, "state": st or "전체", "n": len(dd), "alpha_q": float(f.params["const"]), "r2": float(f.rsquared)}
                for k in fq.columns:
                    row[f"b_{k}"] = float(f.params[k]); row[f"t_{k}"] = float(f.tvalues[k])
                rows.append(row)
    return pd.DataFrame(rows).set_index(["base", "code", "state"])


def rolling_vol(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    return df.rolling(window, min_periods=window).std() * 2


def rolling_betas(df: pd.DataFrame, fq: pd.DataFrame, window: int = 20, min_obs: int = 16) -> pd.DataFrame:
    """분기 롤링 OLS 베타 (창 = window 분기, 창 끝 날짜에 매김). 반환: MultiIndex 컬럼 (code, factor|const|r2)."""
    out = {}
    for c in df.columns:
        d = pd.concat([df[c].rename("y"), fq], axis=1).dropna()
        rows = {}
        for i in range(len(d)):
            w = d.iloc[max(0, i - window + 1): i + 1]
            if len(w) < min_obs:
                continue
            X = np.column_stack([np.ones(len(w)), w[fq.columns].values])
            coef, *_ = np.linalg.lstsq(X, w["y"].values, rcond=None)
            yhat = X @ coef
            r2 = 1 - ((w["y"].values - yhat) ** 2).sum() / ((w["y"].values - w["y"].mean()) ** 2).sum()
            rows[d.index[i]] = dict(zip(["const"] + list(fq.columns), coef)) | {"r2": r2, "n": len(w)}
        if rows:
            out[c] = pd.DataFrame(rows).T
    res = pd.concat(out, axis=1)
    res.index.name = "date"
    return res


def sequential_betas_window(y: np.ndarray, F: np.ndarray, order: list[int]) -> np.ndarray:
    """한 창 안에서 팩터를 order 순서로 Gram-Schmidt 직교화한 뒤 OLS. 반환: 원 팩터 순서의 베타.
    1순위 팩터의 베타 = 단일회귀 베타. k순위 베타 = 앞 팩터들을 뺀 잔차 성분에 대한 베타."""
    n, k = F.shape
    Fc = F - F.mean(0)
    yc = y - y.mean()
    Q = np.zeros_like(Fc)
    for j, idx in enumerate(order):
        v = Fc[:, idx].copy()
        for m in range(j):
            q = Q[:, m]
            v -= (q @ Fc[:, idx]) / (q @ q) * q
        Q[:, j] = v
    b_orth = np.array([(Q[:, j] @ yc) / (Q[:, j] @ Q[:, j]) for j in range(k)])
    out = np.full(k, np.nan)
    for j, idx in enumerate(order):
        out[idx] = b_orth[j]
    return out


def sequential_rolling_betas(df: pd.DataFrame, fq: pd.DataFrame, priority: dict, window: int = 20, min_obs: int = 16) -> pd.DataFrame:
    """시리즈별 우선순위(priority[code] = 팩터 이름 리스트)로 순차 직교화 롤링 베타."""
    out = {}
    cols = list(fq.columns)
    for c in df.columns:
        order = [cols.index(f) for f in priority.get(c, priority.get("default", cols))]
        d = pd.concat([df[c].rename("y"), fq], axis=1).dropna()
        rows = {}
        for i in range(len(d)):
            w = d.iloc[max(0, i - window + 1): i + 1]
            if len(w) < min_obs:
                continue
            b = sequential_betas_window(w["y"].values, w[cols].values, order)
            rows[d.index[i]] = dict(zip(cols, b)) | {"n": len(w)}
        if rows:
            out[c] = pd.DataFrame(rows).T
    res = pd.concat(out, axis=1)
    res.index.name = "date"
    return res
