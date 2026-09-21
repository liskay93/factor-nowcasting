"""확장창 OOS 백테스트 → data/processed/backtest_preds_<code>.csv, backtest_summary.csv, backtest_params_b_<code>.csv"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow.backtest import run_backtest  # noqa: E402
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns, load_factors_daily  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--refit-every", type=int, default=1, help="상태공간 재추정 주기(분기). 1 = 매 분기")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    out = resolve(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg)
    f = load_factors_daily(cfg)
    t0 = time.time()
    res = run_backtest(cfg, alt.series, f, codes=a.codes, refit_every=a.refit_every)
    for code, df in res["preds"].items():
        df.to_csv(out / f"backtest_preds_{code}.csv", float_format="%.8g")
        res["params_b"][code].to_csv(out / f"backtest_params_b_{code}.csv", float_format="%.6g")
    res["summary"].to_csv(out / "backtest_summary.csv", float_format="%.6g")
    print(res["summary"].round(4).to_string())
    print(f"elapsed {time.time() - t0:.1f}s → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
