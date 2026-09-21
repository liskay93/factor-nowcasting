"""보고 수익률 언스무딩 → data/processed/unsmoothed_returns.csv (code), unsmoothed_returns_by_label.csv (엑셀 21 라벨), unsmoothing_params.csv"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns  # noqa: E402
from src.altnow.unsmooth import run_unsmoothing, to_labels  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = resolve(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg)
    r, params = run_unsmoothing(cfg, alt.series)
    r.to_csv(out / "unsmoothed_returns.csv", float_format="%.8g")
    to_labels(r, alt.meta).to_csv(out / "unsmoothed_returns_by_label.csv", float_format="%.8g")
    params.to_csv(out / "unsmoothing_params.csv", float_format="%.6g")
    pd.set_option("display.width", 220)
    print(params[["order", "theta", "sum_theta", "vol_reported_ann", "vol_unsmoothed_ann", "vol_mult",
                  "ac1_reported", "ac1_unsmoothed", "lb4_p_unsmoothed", "n"]].round(3).to_string())
    print(f"saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
