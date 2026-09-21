"""최신 nowcast → data/processed/nowcast_latest.csv (code 별), nowcast_by_label_<model>.csv (엑셀 21 라벨), model_b_params_full.csv, model_a_coef_full.csv"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns, load_factors_daily  # noqa: E402
from src.altnow.nowcast import run_nowcast, to_labels  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    ap.add_argument("--asof", default=None, help="팩터 기준일 (기본: 마지막 관측일)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    out = resolve(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg)
    f = load_factors_daily(cfg)
    res = run_nowcast(cfg, alt.series, f, alt.meta, asof=a.asof or cfg["nowcast"].get("asof"))
    now = res["nowcast"]
    now.to_csv(out / "nowcast_latest.csv", index=False, float_format="%.6g")
    for col in ["ensemble", "ridge_dl", "ss_kalman"]:
        to_labels(now, alt.meta, col).to_csv(out / f"nowcast_by_label_{col}.csv", float_format="%.6g")
    res["params_b"].to_csv(out / "model_b_params_full.csv", float_format="%.6g")
    res["coef_a"].to_csv(out / "model_a_coef_full.csv", float_format="%.6g")
    st = res["status"]
    meta = {"asof": str(st["asof"].date()), "current_quarter_end": str(st["quarter_end"].date()),
            "current_quarter_days": int(st["n_days"]), "current_quarter_complete": st["is_complete"],
            "qtd_factors": {k: float(v) for k, v in st["qtd"].items()}}
    (out / "nowcast_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(now.drop(columns=["index"]).round(4).to_string(index=False))
    print(json.dumps(meta, ensure_ascii=False))
    print(f"saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
