"""대시보드용 테이블.

beta_timeseries (long): 한 행 = (date, program_code, factor, basis, method, window). 프로그램 라벨 21개로 펼친다.
beta_latest (wide):     프로그램 × 4팩터 최신 베타 + R²·n·기준일.
factor_latest:          4팩터 최신 완전분기 수익률, QTD, 연변동성.

DB 키: (date, program_code, factor, basis, method, window_q). 같은 키가 다시 오면 덮어쓴다 (load_db.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import FactorPanels
from .data import quarter_status
from .eda import rolling_betas, sequential_rolling_betas


def build_beta_timeseries(R: dict[str, pd.DataFrame], fq: pd.DataFrame, meta: pd.DataFrame, factors: list[str],
                          methods: list[str], priority: dict, window: int, run_id: str) -> pd.DataFrame:
    frames = []
    for basis, df in R.items():
        for method in methods:
            rb = rolling_betas(df, fq, window=window) if method == "ols" else sequential_rolling_betas(df, fq, priority, window=window)
            for code in rb.columns.get_level_values(0).unique():
                sub = rb[code]
                if code not in meta.index:
                    continue
                m = meta.loc[code]
                for f in factors:
                    if f not in sub.columns:
                        continue
                    s = sub[f].dropna()
                    part = pd.DataFrame({"date": s.index, "beta": s.values})
                    part["r2"] = sub["r2"].reindex(s.index).values if "r2" in sub.columns else np.nan
                    part["n_obs"] = sub["n"].reindex(s.index).values.astype(int)
                    part["program_code"] = code; part["factor"] = f; part["basis"] = basis; part["method"] = method
                    frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    long["window_q"] = window
    long["run_id"] = run_id
    # 프로그램 라벨(21)로 펼치기: 같은 원지수를 쓰는 라벨은 같은 행이 복제된다
    lab = meta.reset_index()[["code", "group", "index", "labels"]].explode("labels").rename(columns={"code": "program_code", "labels": "program", "index": "index_name"})
    long = long.merge(lab, on="program_code", how="left")
    cols = ["date", "program", "program_code", "group", "index_name", "factor", "basis", "method", "window_q", "beta", "r2", "n_obs", "run_id"]
    return long[cols].sort_values(["basis", "method", "program", "factor", "date"]).reset_index(drop=True)


def build_beta_latest(long: pd.DataFrame, factors: list[str], basis: str, method: str) -> pd.DataFrame:
    sub = long[(long.basis == basis) & (long.method == method)]
    last = sub.sort_values("date").groupby(["program", "factor"]).tail(1)
    wide = last.pivot(index="program", columns="factor", values="beta")[factors]
    info = last.groupby("program").agg(asof_quarter=("date", "max"), r2=("r2", "last"), n_obs=("n_obs", "last"),
                                       program_code=("program_code", "first"), group=("group", "first"), index_name=("index_name", "first"))
    out = pd.concat([info, wide], axis=1)
    out["basis"] = basis; out["method"] = method; out["window_q"] = int(sub["window_q"].iloc[0])
    return out.reset_index()


def build_factor_latest(panels: FactorPanels, f_daily: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    st = quarter_status(f_daily)
    fq = panels.fq_full[factors]
    last_q = fq.index[-1]
    rows = []
    for f in factors:
        rows.append({"factor": f, "last_full_quarter": last_q.date(), "ret_last_quarter": float(fq[f].iloc[-1]),
                     "ret_4q": float((1 + fq[f].iloc[-4:]).prod() - 1), "qtd": float(st["qtd"][f]),
                     "qtd_asof": st["asof"].date(), "qtd_quarter": st["quarter_end"].date(), "qtd_days": int(st["n_days"]),
                     "ann_vol_5y": float(fq[f].iloc[-20:].std() * 2), "ann_vol_full": float(fq[f].std() * 2)})
    return pd.DataFrame(rows)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
