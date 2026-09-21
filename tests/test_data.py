"""데이터 계층: 집계 정합성 — 분기 복리 수익률이 factor_levels.csv 분기말 레벨 비율과 (다른 경로로) 일치해야 한다."""
import numpy as np
import pandas as pd
import pytest

from src.altnow.config import load_config, macro_factor_root
from src.altnow.data import (load_alt_returns, load_factors_daily, period_coverage, quarter_status,
                             to_monthly, to_quarterly)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def daily(cfg):
    return load_factors_daily(cfg)


def test_alt_returns_shape(cfg):
    alt = load_alt_returns(cfg)
    assert alt.wide.shape[1] == 21
    assert alt.series.shape[1] == len(cfg["series"])
    assert all(d == d + pd.offsets.QuarterEnd(0) for d in alt.series.index)
    # 라벨이 같은 원지수를 공유하면 값이 같다
    assert np.allclose(alt.wide["Growth"].dropna(), alt.wide["Venture Capital"].dropna())


def test_quarterly_matches_levels(cfg, daily):
    lv = pd.read_csv(macro_factor_root(cfg) / "data/processed/factor_levels.csv", index_col=0, parse_dates=True)
    lq = lv.resample("QE").last()
    lq.index = lq.index + pd.offsets.QuarterEnd(0)
    ratio = (lq / lq.shift(1) - 1).dropna()
    fq = to_quarterly(daily)
    common = ratio.index.intersection(fq.index)
    assert len(common) > 60
    assert (ratio.loc[common] - fq.loc[common]).abs().max().max() < 1e-6      # 레벨은 소수 6자리 저장


def test_monthly_compounds_to_quarterly(daily):
    fm, fq = to_monthly(daily), to_quarterly(daily)
    # 3개월 복리 = 분기 복리 (완전 분기)
    cov = period_coverage(daily, "QE")
    full = cov.index[~cov["partial"]]
    for q in full[5:8]:
        m3 = fm.loc[q - pd.offsets.MonthEnd(2): q]
        assert np.allclose((1 + m3).prod() - 1, fq.loc[q], atol=1e-12)


def test_partial_period_flags(daily):
    cov = period_coverage(daily, "QE")
    assert bool(cov.loc["2006-03-31", "partial"])      # 2006-02-01 시작
    assert not bool(cov.loc["2006-06-30", "partial"])
    st = quarter_status(daily)
    assert st["quarter_start"].day == 1 and st["quarter_end"] == st["asof"] + pd.offsets.QuarterEnd(0)
