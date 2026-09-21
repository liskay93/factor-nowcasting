"""최신 nowcast.

대상 분기 = 각 시리즈의 마지막 보고 분기 이후 ① 팩터가 완전한 분기들(complete) ② 진행 중 분기(partial, QTD).
- 진행 중 분기/부분 월의 팩터: 관측된 QTD 복리 + (1 − 경과비율) × 훈련표본 평균  → "관측 정보 + 남은 기간 무조건 기대치"
- 마지막 보고 분기보다 2분기 이상 앞이면 y_{q−1} 자리에 직전 nowcast 를 넣는다(iterated, h 표시).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .backtest import FactorPanels
from .data import quarter_status, quarterly_lag_panel
from .models.ridge_dl import RidgeDL, build_design
from .models.statespace import fit_ss, monthly_frame

log = logging.getLogger(__name__)
DAYS_PER_MONTH, DAYS_PER_QUARTER = 21, 63


def _fill_partial(fp: pd.DataFrame, cov: pd.DataFrame, full_mean: pd.Series, days_full: int) -> pd.DataFrame:
    """부분 기간 행을 QTD/MTD + (1 − n/days_full) × 평균 으로 보정한다."""
    out = fp.copy()
    for t in out.index:
        if t in cov.index and bool(cov.loc[t, "partial"]):
            frac = min(float(cov.loc[t, "n"]) / days_full, 1.0)
            out.loc[t] = fp.loc[t] + (1.0 - frac) * full_mean
    return out


def run_nowcast(cfg: dict, series: pd.DataFrame, f_daily: pd.DataFrame, meta: pd.DataFrame,
                asof: pd.Timestamp | None = None) -> dict:
    panels = FactorPanels.from_daily(f_daily)
    st = quarter_status(f_daily, asof)
    fq_now = _fill_partial(panels.fq_all, panels.covq, panels.fq_full.mean(), DAYS_PER_QUARTER)
    fm_now = _fill_partial(panels.fm_all, panels.covm, panels.fm_full.mean(), DAYS_PER_MONTH)
    # 첫 분기(2006Q1)는 부분이라 훈련에서 빠지지만 nowcast 패널에는 보정값으로 남긴다 (lag 로 쓰일 일 없음)
    cur_q = None if st["is_complete"] else st["quarter_end"]
    ma, mb = cfg["model_a"], cfg["model_b"]

    rows, params_b, coef_a = [], {}, {}
    for code in series.columns:
        y = series[code].dropna()
        last_y = y.index[-1]
        targets = [q for q in panels.fq_full.index if q > last_y]
        if cur_q is not None and cur_q not in targets:
            targets.append(cur_q)
        if not targets:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ---- 모델 A: 전체 표본 적합 후 순차 예측
            X, yy = build_design(panels.fq_full, y, ma["lags"], ma.get("include_ar", True))
            A = RidgeDL(alphas=ma["alphas"], cv_min_train=ma["cv_min_train"], standardize=ma.get("standardize", True)).fit(X, yy)
            coef_a[code] = A.coef_unscaled()
            lagp = quarterly_lag_panel(fq_now, ma["lags"])
            y_ext_a = y.copy()
            a_hat = {}
            for q in targets:
                x = lagp.loc[[q]].copy()
                if ma.get("include_ar", True):
                    x["y_l1"] = y_ext_a.get(q - pd.offsets.QuarterEnd(1), np.nan)
                x = x[X.columns]
                a_hat[q] = float(A.predict(x)[0]) if not x.isna().any().any() else np.nan
                y_ext_a[q] = a_hat[q]
            # ---- 모델 B: 전체 표본 MLE → 대상 분기까지 필터 → 로그공간에서 순차 결합
            fm_log = np.log1p(fm_now.loc[:targets[-1]])
            fm_train_log = np.log1p(panels.fm_full)
            endog, F, ylag = monthly_frame(fm_log, np.log1p(y), fill_mean=fm_train_log.mean())
            B = fit_ss(endog, F, ylag, theta_bounds=tuple(mb.get("theta_bounds", (0.0, 0.95))))
            params_b[code] = B.params_table()
            y_ext_b = np.log1p(y).copy()
            b_hat, b_sd = {}, {}
            for q in targets:
                i = B.months.get_loc(q)
                c_hat = float(B.res.predicted_state[0, i]); c_var = float(B.res.predicted_state_cov[0, 0, i])
                prev = y_ext_b.get(q - pd.offsets.QuarterEnd(1), np.nan)
                mu = B.theta * prev + (1 - B.theta) * c_hat
                b_hat[q] = float(np.expm1(mu)); b_sd[q] = float(np.sqrt((1 - B.theta) ** 2 * c_var))
                y_ext_b[q] = mu
        for h, q in enumerate(targets, start=1):
            partial = (q == cur_q)
            rows.append({"code": code, "group": meta.loc[code, "group"], "index": meta.loc[code, "index"],
                         "quarter": q, "last_reported": last_y, "h": h,
                         "type": "partial(QTD)" if partial else "complete",
                         "factor_days": int(panels.covq.loc[q, "n"]) if q in panels.covq.index else 0,
                         "ridge_dl": a_hat[q], "ss_kalman": b_hat[q], "ss_sd": b_sd[q],
                         "ensemble": 0.5 * (a_hat[q] + b_hat[q])})
    now = pd.DataFrame(rows)
    return {"nowcast": now, "params_b": pd.DataFrame(params_b).T, "coef_a": pd.DataFrame(coef_a).T,
            "status": st, "panels": panels, "fq_now": fq_now, "fm_now": fm_now}


def to_labels(now: pd.DataFrame, meta: pd.DataFrame, col: str = "ensemble") -> pd.DataFrame:
    """code 별 nowcast → 엑셀 21 라벨(분기 × 라벨)."""
    out = {}
    for code, m in meta.iterrows():
        sub = now[now.code == code].set_index("quarter")[col]
        for lab in m["labels"]:
            out[lab] = sub
    return pd.DataFrame(out)
