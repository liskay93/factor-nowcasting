"""macro_factor 저장소의 레짐 정의(config/regimes.yaml, src/regimes/build.py)를 **그대로 불러** 월간 레짐 라벨을 내보낸다.
macro_factor 는 읽기만 한다 (sys.path 로 import, 캐시 offline). 출력: data/processed/regimes_monthly.csv"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parents[1]
MF = Path(os.environ.get("MACRO_FACTOR_ROOT", HERE.parent / "macro_factor")).resolve()
sys.path.insert(0, str(MF))                      # macro_factor 의 src 패키지를 쓴다 (이 저장소의 src 와 이름이 겹치므로 먼저 넣는다)
for k in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[k]
from src.regimes import build as rb  # noqa: E402  (macro_factor)


def main() -> int:
    cfg = yaml.safe_load((MF / "config/regimes.yaml").read_text(encoding="utf-8"))
    cache = MF / "data/raw/auto"
    out = {}
    out["R1_경기"] = rb.build_R1(cfg, cache, offline=True).series
    out["R3_통화정책"] = rb.build_R3(cfg, cache, offline=True).series
    out["R4_위험선호"] = rb.build_R4(cfg, cache, offline=True).series
    out["R5_서사"] = rb.build_R5(cfg).series
    sel = cfg["R2"]["selected"]
    cand = next(c for c in rb.r2_candidates(cfg) if c["key"] == sel)
    r2 = rb.build_R2_candidate(cand, cache, offline=True)
    out["R2_성장인플레"] = r2.series
    df = pd.DataFrame(out)
    df.index = df.index.to_timestamp(how="end").normalize()
    df.index.name = "month"
    df = df.loc["1999-01-01":]
    dst = HERE / "data/processed/regimes_monthly.csv"
    df.to_csv(dst, encoding="utf-8")
    print(f"R2 selected = {sel}  (vix cut {rb.build_R4(cfg, cache, offline=True).meta})")
    for c in df.columns:
        print(c, df[c].dropna().index[0].date(), "~", df[c].dropna().index[-1].date(), df[c].value_counts().to_dict())
    print(f"saved → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
