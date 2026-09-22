"""베타 행렬 → 포트폴리오 팩터 비중.
출력: data/processed/beta_matrix.csv, beta_matrix_info.csv, factor_weights_<포트폴리오>.csv, factor_weights_ts_<포트폴리오>.csv,
      reports/factor_weights.md, reports/figs/fw_*.png"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow import beta_matrix as bmx, eda_plots  # noqa: E402
from src.altnow.backtest import FactorPanels  # noqa: E402
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns, load_factors_daily  # noqa: E402
from src.altnow.unsmooth import run_unsmoothing  # noqa: E402


def pct(x, d=1):
    return "" if pd.isna(x) else f"{100 * x:.{d}f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    ap.add_argument("--portfolio-config", default="config/portfolio.yaml")
    ap.add_argument("--no-rolling", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    pcfg = yaml.safe_load(resolve(a.portfolio_config).read_text(encoding="utf-8"))
    bm = pcfg["beta_matrix"]
    proc, rep = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["reports"])
    figs = rep / "figs"; figs.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg); factors = cfg["factors"]
    fq = FactorPanels.from_daily(load_factors_daily(cfg)).fq_full
    uns, _ = run_unsmoothing(cfg, alt.series)
    R = uns if bm.get("basis", "unsmoothed") == "unsmoothed" else alt.series
    pri = cfg.get("beta_priority") or {}
    priority = {"default": pri.get("default", factors)} | (pri.get("series") or {})
    rq = pd.read_csv(proc / "regimes_quarterly.csv", index_col=0, parse_dates=True)["R1_경기"] if (proc / "regimes_quarterly.csv").exists() else None

    B, D, info = bmx.estimate_betas(R, fq, bm, priority)
    B.to_csv(proc / "beta_matrix.csv", float_format="%.5g"); info.to_csv(proc / "beta_matrix_info.csv", float_format="%.5g")
    pd.set_option("display.width", 220)
    print(f"베타 행렬 ({bm.get('basis')}, {bm.get('method')}, λ={bm.get('lambda')}, block={bm.get('block')} n_block={bm.get('n_block')})")
    print(pd.concat([B.round(3), info[["lambda", "r2", "n"]].round(3)], axis=1).to_string())

    L = ["# 포트폴리오 팩터 비중\n",
         f"기준: {bm.get('basis')} 수익률 · {bm.get('method')} (λ {bm.get('lambda')}) · 블록 제약 {bm.get('block')} (앞 {bm.get('n_block')}개) · 팩터 Ω 는 분기 공분산 연율화.\n",
         "정의: **팩터 비중 x = B′w** (각 팩터 포트폴리오의 명목 비중, 합 ≠ 1) · 1σ 손익 = x·σ_F (팩터 간 비교) · RC = 오일러 위험기여 (참고).\n",
         "## 베타 행렬 B\n", "| 시리즈 | " + " | ".join(factors) + " | λ | R² | n | 허용 |", "|---|" + "---|" * (len(factors) + 4)]
    for c in B.index:
        L.append(f"| {c} | " + " | ".join(f"{B.loc[c, f]:.2f}" for f in factors) + f" | {info.loc[c, 'lambda']:.3g} | {info.loc[c, 'r2']:.2f} | {int(info.loc[c, 'n'])} | {info.loc[c, 'allowed']} |")
    for pname, p in pcfg["portfolios"].items():
        wl, wc = bmx.weights_to_codes(p["weights"], alt.meta, list(alt.wide.columns))
        res = bmx.factor_weights(B, D, wc, fq)
        s = res["summary"]
        s.to_csv(proc / f"factor_weights_{pname}.csv", float_format="%.5g")
        res["contrib"].assign(w=res["w_code"]).to_csv(proc / f"factor_weights_contrib_{pname}.csv", float_format="%.5g")
        eda_plots.plot_factor_weight_stack(res["contrib"], res["x"], figs / f"fw_stack_{pname}.png")
        eda_plots.plot_factor_summary(s, res["rc_resid_pct"], figs / f"fw_summary_{pname}.png")
        print(f"\n== {p['name']} ==  σ_p={res['sigma_p']:.4f}  고유 RC {res['rc_resid_pct']:.1f}%  검산 {res['check']:.4f}")
        print(s.round(4).to_string())
        L += [f"\n## 포트폴리오: {p['name']}" + (" ⚠가안" if p.get("provisional") else "") + "\n",
              "| 팩터 | **팩터 비중 x** | σ_F | 1σ 손익 | RC | RC% |", "|---|---|---|---|---|---|"]
        for f in factors:
            r = s.loc[f]
            L.append(f"| {f} | **{r.factor_weight_x:.3f}** | {pct(r.sigma_F)} | {pct(r.pnl_1sigma)} | {pct(r.RC, 2)} | {r.RC_pct:.1f}% |")
        L.append(f"| 고유 | | | | {pct(res['rc_resid'], 2)} | {res['rc_resid_pct']:.1f}% |")
        L.append(f"\nσ_p = {pct(res['sigma_p'])}. 자산별 기여 w_i·β_i:\n")
        L.append("| 시리즈 | w | " + " | ".join(factors) + " |"); L.append("|---|---|" + "---|" * len(factors))
        for c in res["contrib"].index:
            L.append(f"| {c} | {pct(res['w_code'][c])} | " + " | ".join(f"{res['contrib'].loc[c, f]:.3f}" for f in factors) + " |")
        L.append(f"\n![stack](figs/fw_stack_{pname}.png)\n\n![summary](figs/fw_summary_{pname}.png)\n")
        if not a.no_rolling:
            lam = float(info["lambda"].median())
            xt = bmx.rolling_factor_weights(R, fq, bm, priority, wc, window=int(bm.get("rolling_window", 20)), lam=lam)
            if len(xt):
                xt.to_csv(proc / f"factor_weights_ts_{pname}.csv", float_format="%.5g")
                eda_plots.plot_factor_weight_ts(xt, figs / f"fw_ts_{pname}.png", rq)
                L.append(f"롤링 {bm.get('rolling_window', 20)}분기 베타로 본 x_t (λ={lam:.3g} 고정):\n\n![ts](figs/fw_ts_{pname}.png)\n")
    L += ["\n## 주의", "- x 는 명목 노출이라 팩터 간 크기 비교는 1σ 손익·RC 로 한다.",
          "- 블록 제약·ridge 축소는 사람이 정한 것이다 (config/portfolio.yaml, nowcast.yaml beta_priority). 바꾸면 x 가 바뀐다.",
          "- 감정평가 부동산(MSCI GPFI·Burgiss RE)은 동분기 베타가 lag 를 못 잡아 과소일 수 있다."]
    (rep / "factor_weights.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nsaved → {rep / 'factor_weights.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
