"""대시보드용 테이블 내보내기 → data/dashboard/{beta_timeseries.csv, beta_latest.csv, factor_latest.csv, meta.json} (+ parquet)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow import dashboard as dash  # noqa: E402
from src.altnow.backtest import FactorPanels  # noqa: E402
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns, load_factors_daily  # noqa: E402
from src.altnow.unsmooth import run_unsmoothing  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    ap.add_argument("--out", default="data/dashboard")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); dc = cfg["dashboard"]
    out = resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg); factors = cfg["factors"]
    f = load_factors_daily(cfg); panels = FactorPanels.from_daily(f); fq = panels.fq_full
    uns, uparams = run_unsmoothing(cfg, alt.series)
    R = {b: (alt.series if b == "reported" else uns) for b in dc["basis"]}
    pri = cfg.get("beta_priority") or {}
    priority = {"default": pri.get("default", factors)} | (pri.get("series") or {})
    run_id = dash.new_run_id()

    long = dash.build_beta_timeseries(R, fq, alt.meta, factors, dc["methods"], priority, int(dc["window"]), run_id)
    latest = dash.build_beta_latest(long, factors, dc["table_basis"], dc["table_method"])
    flat = dash.build_factor_latest(panels, f, factors)
    long.to_csv(out / "beta_timeseries.csv", index=False, float_format="%.6g")
    latest.to_csv(out / "beta_latest.csv", index=False, float_format="%.6g")
    flat.to_csv(out / "factor_latest.csv", index=False, float_format="%.6g")
    try:
        long.to_parquet(out / "beta_timeseries.parquet", index=False)
    except Exception as e:  # pyarrow 없으면 CSV 만
        print(f"(parquet 생략: {e})")
    meta = {"run_id": run_id, "factor_asof": str(f.index[-1].date()), "last_full_quarter": str(fq.index[-1].date()),
            "alt_last_quarter": str(alt.series.index[-1].date()), "basis": dc["basis"], "methods": dc["methods"], "window_q": int(dc["window"]),
            "table_basis": dc["table_basis"], "table_method": dc["table_method"],
            "unsmoothing": {c: int(uparams.loc[c, "order"]) for c in uparams.index},
            "theta_mode": (cfg.get("unsmoothing") or {}).get("theta_mode", "constant"),
            "n_rows_timeseries": int(len(long)), "programs": sorted(long["program"].unique().tolist())}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(f"beta_timeseries: {len(long)} rows  {long.date.min().date()}~{long.date.max().date()}  basis {dc['basis']} methods {dc['methods']}")
    print(long.head(3).to_string(index=False))
    print("\nbeta_latest:"); print(latest[["program", "asof_quarter"] + factors + ["r2"]].round(3).to_string(index=False))
    print("\nfactor_latest:"); print(flat.round(4).to_string(index=False))
    print(f"\nsaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
