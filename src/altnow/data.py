"""데이터 계층.

- 대체투자 분기 수익률: 엑셀 `값` 시트 (행1 그룹, 행2 라벨, 행3 원지수명, 행4~ 분기말 날짜 + 단순수익률).
  원지수가 같은 라벨은 하나의 시리즈(code)로 모델링하고 결과는 라벨로 되돌린다.
- 매크로 팩터: macro_factor 저장소의 `data/processed/factors.csv` (일간 단순수익률). **읽기만 한다.**
  일간 → 월간·분기는 복리 Π(1+r)−1 (macro_factor 의 규약과 같다).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import macro_factor_root, resolve

log = logging.getLogger(__name__)


@dataclass
class AltData:
    wide: pd.DataFrame       # 분기말 × 라벨(21). 엑셀 그대로
    series: pd.DataFrame     # 분기말 × code(15). 고유 원지수
    meta: pd.DataFrame       # code, index, group, labels


def load_alt_returns(cfg: dict) -> AltData:
    path = resolve(cfg["paths"]["alt_returns"])
    raw = pd.read_excel(path, sheet_name=cfg["paths"].get("alt_sheet", 0), header=None)
    labels = raw.iloc[1, 1:].astype(str).str.strip().tolist()
    index_names = raw.iloc[2, 1:].astype(str).str.strip().tolist()
    body = raw.iloc[3:].copy()
    body.columns = ["date"] + labels
    body["date"] = pd.to_datetime(body["date"])
    body = body.set_index("date").sort_index()
    body = body.apply(pd.to_numeric, errors="coerce")       # '#N/A' 등 → NaN
    body.index.name = "date"
    # 분기말 정렬 확인
    bad = [d for d in body.index if d != (d + pd.offsets.QuarterEnd(0))]
    if bad:
        raise ValueError(f"분기말이 아닌 날짜: {bad[:5]}")

    # 라벨 → 시리즈 code 매핑 (config.series)
    label2code, rows = {}, []
    for code, s in cfg["series"].items():
        for lab in s["labels"]:
            label2code[lab] = code
        rows.append({"code": code, "index": s["index"], "group": s["group"], "labels": list(s["labels"])})
    meta = pd.DataFrame(rows).set_index("code")
    unmapped = [l for l in labels if l not in label2code]
    if unmapped:
        raise KeyError(f"config.series 에 없는 엑셀 라벨: {unmapped}")

    # 원지수명이 config 와 맞는지, 같은 code 라벨끼리 값이 같은지 확인
    series = {}
    for code, s in cfg["series"].items():
        cols = [l for l in s["labels"] if l in body.columns]
        for l in cols:
            j = labels.index(l)
            if index_names[j] != s["index"]:
                log.warning("라벨 %s 의 원지수명 '%s' ≠ config '%s'", l, index_names[j], s["index"])
        block = body[cols]
        if len(cols) > 1:
            ref = block.iloc[:, 0]
            for l in cols[1:]:
                diff = (block[l] - ref).abs().max()
                if not (np.isnan(diff) or diff < 1e-12):
                    log.warning("같은 원지수 라벨 %s vs %s 값이 다름 (max|diff|=%.3g) — 첫 라벨 사용", cols[0], l, diff)
        series[code] = block.iloc[:, 0]
    series = pd.DataFrame(series)
    series.index.name = "date"
    log.info("alt returns: %d분기 %s~%s, 라벨 %d → 시리즈 %d",
             len(series), series.index[0].date(), series.index[-1].date(), len(labels), series.shape[1])
    return AltData(wide=body, series=series, meta=meta)


def load_factors_daily(cfg: dict) -> pd.DataFrame:
    root = macro_factor_root(cfg)
    path = root / cfg["paths"]["factor_file"]
    if not path.exists():
        raise FileNotFoundError(f"팩터 파일 없음: {path} (MACRO_FACTOR_ROOT 확인)")
    f = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    f.index.name = "date"
    cols = cfg.get("factors") or list(f.columns)
    f = f[cols]
    log.info("factors: %s  %s~%s  %d일", path, f.index[0].date(), f.index[-1].date(), len(f))
    return f


def compound(r: pd.DataFrame, rule: str) -> pd.DataFrame:
    """일간 단순수익률 → 기간 복리 수익률. 결측은 곱에서 제외(그날 관측 없음). 관측이 하나도 없는 기간은 NaN."""
    g = r.resample(rule)
    out = g.apply(lambda x: np.prod(1.0 + x.dropna().values) - 1.0 if x.notna().any() else np.nan)
    return out


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    m = compound(daily, "ME")
    m.index = m.index + pd.offsets.MonthEnd(0)
    return m


def to_quarterly(daily: pd.DataFrame) -> pd.DataFrame:
    q = compound(daily, "QE")
    q.index = q.index + pd.offsets.QuarterEnd(0)
    return q


def period_coverage(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """기간별 첫/마지막 관측일·관측 수와 부분기간 여부. 첫 관측이 기간 시작 +7일 이후이거나
    마지막 관측이 기간 끝 −7일 이전이면 부분기간(partial)으로 본다."""
    idx = pd.Series(daily.index, index=daily.index)
    g = idx.resample(rule)
    cov = pd.DataFrame({"first": g.min(), "last": g.max(), "n": g.count()})
    cov = cov[cov["n"] > 0]
    off = pd.offsets.QuarterEnd(0) if rule.startswith("Q") else pd.offsets.MonthEnd(0)
    cov.index = cov.index + off
    pstart = pd.DatetimeIndex([pd.Timestamp(t.year, t.month - (2 if rule.startswith("Q") else 0), 1) for t in cov.index])
    cov["partial"] = (cov["first"] > pstart + pd.Timedelta(days=7)) | (cov["last"] < cov.index - pd.Timedelta(days=7))
    return cov


def quarter_status(daily: pd.DataFrame, asof: pd.Timestamp | None = None) -> dict:
    """마지막 관측 기준 진행 중 분기의 상태: 분기말, 경과 영업일, 완료 월 수, QTD 복리 수익률."""
    asof = pd.Timestamp(asof) if asof is not None else daily.index[-1]
    d = daily.loc[:asof]
    qend = asof + pd.offsets.QuarterEnd(0)
    qstart = pd.Timestamp(qend.year, qend.month - 2, 1)
    cur = d.loc[qstart:asof]
    is_complete = bool(asof >= qend - pd.Timedelta(days=3))
    months_done = (asof.month - qstart.month) + (1 if asof >= asof + pd.offsets.MonthEnd(0) - pd.Timedelta(days=3) else 0)
    qtd = (1.0 + cur).prod() - 1.0
    return {"asof": asof, "quarter_end": qend, "quarter_start": qstart, "n_days": len(cur),
            "months_complete": int(months_done), "is_complete": is_complete, "qtd": qtd}


def quarterly_lag_panel(fq: pd.DataFrame, lags: int) -> pd.DataFrame:
    """분기 팩터 → lag 0..L 패널. 컬럼명 `<factor>_l<k>`."""
    blocks = {f"{c}_l{k}": fq[c].shift(k) for k in range(lags + 1) for c in fq.columns}
    return pd.DataFrame(blocks, index=fq.index)
