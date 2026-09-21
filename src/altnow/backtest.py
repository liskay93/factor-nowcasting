"""확장창 pseudo-real-time 백테스트.

분기 q 를 예측할 때 쓰는 정보: 팩터는 q 분기말까지(완전 분기), 대체투자 보고수익률은 q−1 까지.
주의(의사 실시간): ① Burgiss 류 지수는 실제로는 1~2분기 늦게 발표되고 개정되지만 여기서는 최종 빈티지를 쓴다.
② y_{q−1} 을 q 분기말에 안다고 가정한다. 발표 지연을 반영하려면 target_known_lag 를 2 로 두면
   AR 항과 상태공간의 y_{q−1} 자리에 y_{q−2} 가 아닌 '관측 안 함' 처리를 해야 하므로 여기서는 옵션으로만 남긴다.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import period_coverage, to_monthly, to_quarterly
from .models.ridge_dl import RidgeDL, build_design
from .models.statespace import fit_ss, monthly_frame

log = logging.getLogger(__name__)


@dataclass
class FactorPanels:
    fq_full: pd.DataFrame     # 완전 분기 팩터(단순수익률)
    fm_full: pd.DataFrame     # 완전 월 팩터(단순수익률)
    fq_all: pd.DataFrame      # 부분 분기 포함
    fm_all: pd.DataFrame
    covq: pd.DataFrame
    covm: pd.DataFrame

    @classmethod
    def from_daily(cls, f: pd.DataFrame) -> "FactorPanels":
        covq, covm = period_coverage(f, "QE"), period_coverage(f, "ME")
        fq, fm = to_quarterly(f), to_monthly(f)
        fq_full = fq[~covq["partial"].reindex(fq.index).fillna(True).astype(bool)]
        fm_full = fm[~covm["partial"].reindex(fm.index).fillna(True).astype(bool)]
        return cls(fq_full, fm_full, fq, fm, covq, covm)


def predict_mean(y_train: pd.Series) -> float:
    return float(y_train.mean())


def predict_ar1(y_train: pd.Series, y_last: float) -> float:
    yl = y_train.shift(1).dropna()
    yy = y_train.loc[yl.index]
    X = np.column_stack([np.ones(len(yl)), yl.values])
    coef, *_ = np.linalg.lstsq(X, yy.values, rcond=None)
    return float(coef[0] + coef[1] * y_last)


def predict_ols_contemp(fq_full: pd.DataFrame, y_train: pd.Series, q: pd.Timestamp) -> float:
    """가장 단순한 팩터 벤치마크: y_q = c + φ y_{q-1} + b'F_q (동분기 팩터만, 벌점 없음).
    모델 B 의 점예측은 완전 분기에서 이 식의 제약형(로그수익률, b = (1-θ)β)이라 거의 같은 값이 나온다 —
    B 의 추가 가치는 부분 분기 처리·θ/β 분해·표준편차이지 완전 분기 점예측이 아니다."""
    df = pd.concat([y_train.rename("y"), y_train.shift(1).rename("yl"), fq_full], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["yl"].values, df[fq_full.columns].values])
    coef, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    if q not in fq_full.index:
        return np.nan
    x = np.r_[1.0, float(y_train.iloc[-1]), fq_full.loc[q].values]
    return float(x @ coef)


def predict_ridge(fq_full: pd.DataFrame, y_train: pd.Series, q: pd.Timestamp, mcfg: dict) -> tuple[float, RidgeDL]:
    """q 분기 예측. 훈련은 y_train (q 이전) 과 그때까지의 팩터."""
    y_ext = pd.concat([y_train, pd.Series([np.nan], index=[q])])       # q 행을 만들기 위한 자리
    from .data import quarterly_lag_panel
    X = quarterly_lag_panel(fq_full.loc[:q], mcfg["lags"])
    if mcfg.get("include_ar", True):
        X["y_l1"] = y_ext.shift(1)
    df = pd.concat([X, y_ext.rename("y")], axis=1)
    train = df.loc[:q - pd.Timedelta(days=1)].dropna()
    xq = df.loc[[q]].drop(columns="y")
    if xq.isna().any().any():
        return np.nan, None
    m = RidgeDL(alphas=mcfg["alphas"], cv_min_train=mcfg["cv_min_train"], standardize=mcfg.get("standardize", True))
    m.fit(train.drop(columns="y"), train["y"])
    return float(m.predict(xq)[0]), m


def predict_ss(fm_full: pd.DataFrame, y_train: pd.Series, q: pd.Timestamp, mcfg: dict,
               start_params=None, fill_mean: pd.Series | None = None) -> tuple[float, float, object]:
    """q 분기 nowcast (로그 → 단순). 훈련 관측은 y_train(q 이전), 팩터는 q 까지의 월."""
    fm_log = np.log1p(fm_full.loc[:q])
    yq_log = np.log1p(y_train)
    fill = fill_mean if fill_mean is not None else fm_log.mean()
    endog, F, ylag = monthly_frame(fm_log, yq_log, fill_mean=fill)
    if q not in endog.index:
        return np.nan, np.nan, None
    # q 의 y_{q-1} 는 관측식 절편에 필요 — monthly_frame 은 y_q 가 없으면 ylag 도 0 으로 두므로 여기서 채운다
    prev = q - pd.offsets.QuarterEnd(1)
    if prev not in yq_log.index:
        return np.nan, np.nan, None
    ylag[q] = yq_log[prev]
    r = fit_ss(endog, F, ylag, theta_bounds=tuple(mcfg.get("theta_bounds", (0.0, 0.95))), start_params=start_params)
    mu, sd = r.nowcast(q)
    return float(np.expm1(mu)), float(sd), r


def run_backtest(cfg: dict, series: pd.DataFrame, f_daily: pd.DataFrame, codes: list[str] | None = None,
                 refit_every: int = 1) -> dict:
    """반환: {'preds': {code: DataFrame(q × model)}, 'summary': DataFrame(code, model × 지표), 'params_b': {code: DataFrame}}"""
    bt = cfg["backtest"]
    panels = FactorPanels.from_daily(f_daily)
    start = pd.Timestamp(bt["start"])
    codes = codes or list(series.columns)
    preds, params_b, ridge_alpha = {}, {}, {}
    for code in codes:
        y = series[code].dropna()
        targets = [q for q in y.index if q >= start and q in panels.fq_full.index]
        rows, prow, sp = [], [], None
        for i, q in enumerate(targets):
            y_train = y.loc[:q - pd.Timedelta(days=1)]
            if len(y_train) < bt["min_train_quarters"]:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a_hat, m_a = predict_ridge(panels.fq_full, y_train, q, cfg["model_a"])
                sp_use = sp if (sp is not None and i % refit_every != 0) else None
                b_hat, b_sd, r_b = predict_ss(panels.fm_full, y_train, q, cfg["model_b"], start_params=sp_use)
            if r_b is not None:
                sp = r_b.res.params
                pt = r_b.params_table(); pt.name = q; prow.append(pt)
            rows.append({"date": q, "y": float(y[q]), "mean": predict_mean(y_train),
                         "ar1": predict_ar1(y_train, float(y_train.iloc[-1])),
                         "ols": predict_ols_contemp(panels.fq_full, y_train, q),
                         "ridge_dl": a_hat, "ss_kalman": b_hat, "ss_sd": b_sd,
                         "ridge_alpha": (m_a.alpha_ if m_a is not None else np.nan)})
        if not rows:
            log.warning("%s: 백테스트 표본 부족", code)
            continue
        df = pd.DataFrame(rows).set_index("date")
        df["ensemble"] = 0.5 * (df["ridge_dl"] + df["ss_kalman"])      # 두 모델 단순평균
        preds[code] = df
        params_b[code] = pd.DataFrame(prow)
        log.info("%s: %d분기 OOS  RMSE mean %.4f ar1 %.4f ridge %.4f ss %.4f", code, len(df),
                 np.sqrt(((df.y - df["mean"]) ** 2).mean()), np.sqrt(((df.y - df.ar1) ** 2).mean()),
                 np.sqrt(((df.y - df.ridge_dl) ** 2).mean()), np.sqrt(((df.y - df.ss_kalman) ** 2).mean()))
    from .metrics import summarize
    summ = {}
    for code, df in preds.items():
        s = summarize(df["y"], df[["mean", "ar1", "ols", "ridge_dl", "ss_kalman", "ensemble"]], bench_col="mean", dm_vs="ar1")
        summ[code] = s
    summary = pd.concat(summ, names=["code", "model"]) if summ else pd.DataFrame()
    return {"preds": preds, "summary": summary, "params_b": params_b, "panels": panels}
