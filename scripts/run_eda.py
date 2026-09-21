"""보고 vs 언스무딩 EDA → reports/eda_report.md + reports/figs/eda_*.png + data/processed/eda_*.csv"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow import eda, eda_plots  # noqa: E402
from src.altnow.backtest import FactorPanels  # noqa: E402
from src.altnow.config import load_config, macro_factor_root, resolve  # noqa: E402
from src.altnow.data import load_alt_returns, load_factors_daily  # noqa: E402
from src.altnow.unsmooth import run_unsmoothing  # noqa: E402


def pct(x, d=1):
    return "" if pd.isna(x) else f"{100 * x:.{d}f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    proc, rep = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["reports"])
    figs = rep / "figs"; figs.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg)
    order = list(alt.meta.index)
    factors = cfg["factors"]
    uns, uparams = run_unsmoothing(cfg, alt.series)
    R = {"reported": alt.series, "unsmoothed": uns}
    rm = pd.read_csv(proc / "regimes_monthly.csv", index_col=0, parse_dates=True)
    rq = eda.regimes_quarterly(rm)
    rq.to_csv(proc / "regimes_quarterly.csv", encoding="utf-8")
    dgs = pd.read_csv(macro_factor_root(cfg) / "data/raw/auto/fred/DGS3MO.csv", index_col=0, parse_dates=True).iloc[:, 0]
    rf = eda.cash_quarterly(dgs)
    fq = FactorPanels.from_daily(load_factors_daily(cfg)).fq_full

    stats = eda.summary_stats(R, rf); stats.to_csv(proc / "eda_summary_stats.csv", float_format="%.6g")
    betas = eda.factor_betas(R, fq); betas.to_csv(proc / "eda_factor_betas.csv", float_format="%.6g")
    betas_r4 = eda.factor_betas(R, fq, rq, "R4_위험선호"); betas_r4.to_csv(proc / "eda_factor_betas_by_risk.csv", float_format="%.6g")
    regime_tabs = {fam: eda.regime_stats(R, rq, fam) for fam in eda.REGIME_ORDER}
    for fam, t in regime_tabs.items():
        t.to_csv(proc / f"eda_regime_{fam.split('_')[0]}.csv", float_format="%.6g", encoding="utf-8")

    # ---- 그림
    eda_plots.plot_stats_dumbbell(stats, order, figs / "eda_stats.png")
    for fam in ["R1_경기", "R2_성장인플레", "R4_위험선호", "R3_통화정책"]:
        eda_plots.plot_regime_heatmaps(regime_tabs[fam], fam, order, figs / f"eda_regime_{fam.split('_')[0]}.png")
    eda_plots.plot_beta_dumbbell(betas, order, factors, figs / "eda_betas.png")
    eda_plots.plot_corr_pair(R, order, figs / "eda_corr.png")
    eda_plots.plot_small_multiples(R, order, figs / "eda_levels.png", "level", rq["R1_경기"])
    eda_plots.plot_small_multiples(R, order, figs / "eda_rollvol.png", "rollvol", rq["R1_경기"])

    # ---- 리포트
    L = ["# 대체투자 수익률 EDA — 보고(스무딩) vs 언스무딩\n",
         f"언스무딩 설정: `config/nowcast.yaml` unsmoothing (차수 " + ", ".join(f"{c} {int(uparams.loc[c, 'order'])}" for c in order) + "). "
         "레짐은 macro_factor 정의(R1 USREC · R2 indpro_yoy|bei10|R1_level · R3 FEDFUNDS 3개월 ±25bp · R4 VIX 상위 20% · R5 서사)를 분기 다수결로 모았다. "
         "연율화: 평균 ×4, 변동성 ×2. 샤프는 3M T-bill 차감.\n"]
    L.append("## 1. 기초 통계\n")
    L.append("| 시리즈 | n | 기간 | 기준 | 연수익(기하) | 연변동성 | 샤프 | 왜도 | 첨도 | 최악 분기 | MDD | AC1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in order:
        for base in ["reported", "unsmoothed"]:
            s = stats.loc[(base, c)]
            L.append(f"| {c if base == 'reported' else ''} | {int(s.n)} | {s.start}~{s.end} | {eda.BASES[base]} | {pct(s.ann_return_geo)} | {pct(s.ann_vol)} | "
                     f"{s['sharpe']:.2f} | {s['skew']:.2f} | {s['kurt']:.1f} | {pct(s['worst_q'])} | {pct(s['max_dd'])} | {s['ac1']:.2f} |")
    L.append("\n![stats](figs/eda_stats.png)\n")
    L.append("## 2. 팩터 베타 (분기, HAC t)\n")
    L.append("| 시리즈 | 기준 | " + " | ".join(f"β {f}" for f in factors) + " | R² |")
    L.append("|---|---|" + "---|" * (len(factors) + 1))
    for c in order:
        for base in ["reported", "unsmoothed"]:
            b = betas.loc[(base, c, "전체")]
            L.append(f"| {c if base == 'reported' else ''} | {eda.BASES[base]} | " + " | ".join(f"{b[f'b_{f}']:.2f} ({b[f't_{f}']:.1f})" for f in factors) + f" | {b.r2:.2f} |")
    L.append("\n![betas](figs/eda_betas.png)\n")
    L.append("리스크온/오프(R4) 별 growth 베타:\n")
    L.append("| 시리즈 | 기준 | β growth 리스크온 | β growth 리스크오프 | R² 온 | R² 오프 |")
    L.append("|---|---|---|---|---|---|")
    for c in order:
        for base in ["reported", "unsmoothed"]:
            on = betas_r4.loc[(base, c, "리스크온")] if (base, c, "리스크온") in betas_r4.index else None
            off = betas_r4.loc[(base, c, "리스크오프")] if (base, c, "리스크오프") in betas_r4.index else None
            L.append(f"| {c if base == 'reported' else ''} | {eda.BASES[base]} | {on.b_growth:.2f} | {off.b_growth:.2f} | {on.r2:.2f} | {off.r2:.2f} |" if on is not None and off is not None else f"| {c} | {eda.BASES[base]} | | | | |")
    for fam in eda.REGIME_ORDER:
        t = regime_tabs[fam]; states = eda.REGIME_ORDER[fam]
        L.append(f"\n## 레짐 {fam} — 연율 평균 / 연율 변동성 (n)\n")
        L.append("| 시리즈 | 기준 | " + " | ".join(states) + " |")
        L.append("|---|---|" + "---|" * len(states))
        for c in order:
            for base in ["reported", "unsmoothed"]:
                cells = []
                for s in states:
                    if (base, c, s) in t.index and not pd.isna(t.loc[(base, c, s)].get("ann_mean", np.nan)):
                        x = t.loc[(base, c, s)]; cells.append(f"{pct(x.ann_mean)} / {pct(x.ann_vol)} ({int(x.n)})")
                    else:
                        cells.append("–")
                L.append(f"| {c if base == 'reported' else ''} | {eda.BASES[base]} | " + " | ".join(cells) + " |")
        if fam != "R5_서사":
            L.append(f"\n![{fam}](figs/eda_regime_{fam.split('_')[0]}.png)\n")
    L.append("\n## 상관·누적·롤링 변동성\n\n![corr](figs/eda_corr.png)\n\n![levels](figs/eda_levels.png)\n\n![rollvol](figs/eda_rollvol.png)\n")
    (rep / "eda_report.md").write_text("\n".join(L), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(stats[["ann_return_geo", "ann_vol", "sharpe", "max_dd", "ac1"]].unstack("base").round(3).to_string())
    print(f"saved → {rep / 'eda_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
