"""분기 롤링 팩터 베타 (보고 vs 언스무딩) → data/processed/rolling_betas_<base>.csv + reports/figs/beta_ts_*.png"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow import eda, eda_plots  # noqa: E402
from src.altnow.backtest import FactorPanels  # noqa: E402
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns, load_factors_daily  # noqa: E402
from src.altnow.unsmooth import run_unsmoothing  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    ap.add_argument("--window", type=int, default=20, help="롤링 창(분기). 기본 20 = 5년")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    proc, figs = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["reports"]) / "figs"
    alt = load_alt_returns(cfg); order = list(alt.meta.index); factors = cfg["factors"]
    fq = FactorPanels.from_daily(load_factors_daily(cfg)).fq_full
    uns, _ = run_unsmoothing(cfg, alt.series)
    rq = pd.read_csv(proc / "regimes_quarterly.csv", index_col=0, parse_dates=True)["R1_경기"] if (proc / "regimes_quarterly.csv").exists() else None
    R = {"reported": alt.series, "unsmoothed": uns}
    rb = {}
    for base, df in R.items():
        rb[base] = eda.rolling_betas(df, fq, window=a.window)
        flat = rb[base].copy(); flat.columns = [f"{c}|{k}" for c, k in flat.columns]
        flat.to_csv(proc / f"rolling_betas_{base}.csv", float_format="%.5g")
        eda_plots.plot_rolling_betas(rb[base], order, factors, figs / f"beta_ts_{base}.png",
                                     f"{a.window}-quarter rolling factor betas — {eda_plots.BASE_NAME[base]}", rq)
    for f in factors:
        eda_plots.plot_rolling_beta_compare(rb["reported"], rb["unsmoothed"], order, f, figs / f"beta_ts_compare_{f}.png", rq)
    # 우선순위(순차 직교화) 베타 — 언스무딩 기준
    pri = cfg.get("beta_priority") or {}
    priority = {"default": pri.get("default", factors)} | (pri.get("series") or {})
    rb_seq = eda.sequential_rolling_betas(uns, fq, priority, window=a.window)
    flat = rb_seq.copy(); flat.columns = [f"{c}|{k}" for c, k in flat.columns]
    flat.to_csv(proc / "rolling_betas_unsmoothed_priority.csv", float_format="%.5g")
    eda_plots.plot_rolling_betas(rb_seq, order, factors, figs / "beta_ts_unsmoothed_priority.png",
                                 f"{a.window}-quarter rolling betas, priority-ordered (sequential orthogonalisation) — unsmoothed", rq)
    for f in factors:
        eda_plots.plot_beta_method_compare(rb["unsmoothed"], rb_seq, order, f, figs / f"beta_ts_method_{f}.png", rq)
    rb["unsmoothed_priority"] = rb_seq
    # 최신 값 표
    pd.set_option("display.width", 220)
    last = {}
    for base in R:
        t = rb[base].iloc[-1].unstack(0).T if False else pd.DataFrame({c: rb[base][c].dropna().iloc[-1] for c in order if c in rb[base].columns.get_level_values(0)}).T
        last[base] = t[factors + (["r2"] if "r2" in t.columns else [])]
    out = pd.concat(last, axis=1)
    print(out.round(2).to_string())
    print(f"saved → {proc}/rolling_betas_*.csv, {figs}/beta_ts_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
