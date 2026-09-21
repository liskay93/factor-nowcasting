"""백테스트·nowcast 산출물(CSV) → reports/figs/*.png + reports/nowcast_report.md"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow import plots  # noqa: E402
from src.altnow.config import load_config, resolve  # noqa: E402
from src.altnow.data import load_alt_returns  # noqa: E402


def pct(x, d=2):
    return "" if pd.isna(x) else f"{100 * x:.{d}f}%"


def qlabel(q: pd.Timestamp) -> str:
    return f"{q.year}Q{(q.month - 1) // 3 + 1}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    proc, rep = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["reports"])
    figs = rep / "figs"; figs.mkdir(parents=True, exist_ok=True)
    alt = load_alt_returns(cfg)
    order = list(alt.meta.index)
    summary = pd.read_csv(proc / "backtest_summary.csv", index_col=[0, 1])
    preds = {c: pd.read_csv(proc / f"backtest_preds_{c}.csv", index_col=0, parse_dates=True) for c in order
             if (proc / f"backtest_preds_{c}.csv").exists()}
    now = pd.read_csv(proc / "nowcast_latest.csv", parse_dates=["quarter", "last_reported"])
    pb = pd.read_csv(proc / "model_b_params_full.csv", index_col=0)
    meta = json.loads((proc / "nowcast_meta.json").read_text(encoding="utf-8"))
    factors = cfg["factors"]

    # ---- 그림
    plots.plot_oos_r2(summary, order, figs / "oos_r2.png")
    plots.plot_oos_paths(preds, [c for c in order if c in preds], figs / "oos_paths.png")
    latest_q = sorted(now.quarter.unique())[-2:]
    plots.plot_latest_nowcast(now, order, figs / "nowcast_latest.png", [pd.Timestamp(q) for q in latest_q])
    plots.plot_theta(pb, order, figs / "theta.png")
    plots.plot_beta_heatmap(pb, order, factors, figs / "beta_economic.png", reported=False)
    plots.plot_beta_heatmap(pb, order, factors, figs / "beta_reported.png", reported=True)

    # ---- 표
    L = []
    L.append("# 대체투자 분기수익률 Nowcast 리포트\n")
    L.append(f"팩터 기준일 **{meta['asof']}** · 진행 중 분기 {qlabel(pd.Timestamp(meta['current_quarter_end']))} "
             f"(경과 영업일 {meta['current_quarter_days']}) · 팩터 = macro_factor `factors.csv` (growth / term / inflation / credit, 2006-02~)\n")
    L.append("QTD 팩터: " + ", ".join(f"{k} {pct(v)}" for k, v in meta["qtd_factors"].items()) + "\n")

    L.append("## 1. 최신 nowcast (시리즈별)\n")
    L.append("| 시리즈 | 그룹 | 분기 | 유형 | h | A: Ridge DL | B: Kalman | B s.d. | **Ensemble** |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in now.iterrows():
        L.append(f"| {r.code} | {r.group} | {qlabel(r.quarter)} | {r.type} | {r.h} | {pct(r.ridge_dl)} | {pct(r.ss_kalman)} | {pct(r.ss_sd)} | **{pct(r.ensemble)}** |")
    L.append("\nh = 마지막 보고 분기 이후 몇 번째 분기인가 (h≥2 는 직전 nowcast 를 y_{q−1} 로 넣은 반복 예측). "
             "partial(QTD) 는 진행 중 분기: 관측된 QTD 팩터 + 남은 기간 무조건 평균.\n")
    L.append("![latest](figs/nowcast_latest.png)\n")

    L.append("## 2. 최신 nowcast (엑셀 21 라벨, Ensemble)\n")
    lab = pd.read_csv(proc / "nowcast_by_label_ensemble.csv", index_col=0, parse_dates=True)
    L.append("| 라벨 | " + " | ".join(qlabel(q) for q in lab.index) + " |")
    L.append("|---|" + "---|" * len(lab.index))
    for c in lab.columns:
        L.append(f"| {c} | " + " | ".join(pct(v) for v in lab[c].values) + " |")
    L.append("")

    L.append("## 3. 백테스트 (확장창, pseudo-real-time OOS)\n")
    bt = cfg["backtest"]
    L.append(f"첫 OOS 분기 {qlabel(pd.Timestamp(bt['start']))}, 최소 훈련 {bt['min_train_quarters']}분기. "
             "매 분기 모든 모델을 그 시점까지의 자료로 다시 추정. 벤치마크: 확장창 평균, AR(1).\n")
    L.append("| 시리즈 | n | RMSE 평균 | RMSE AR(1) | RMSE OLS | RMSE A | RMSE B | **RMSE Ens** | R²_OOS Ens | 적중률 Ens | DM p (Ens vs AR1) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in order:
        if (c, "ensemble") not in summary.index:
            continue
        s = summary.loc[c]
        L.append(f"| {c} | {int(s.loc['ensemble', 'n'])} | {pct(s.loc['mean', 'rmse'])} | {pct(s.loc['ar1', 'rmse'])} | {pct(s.loc['ols', 'rmse'])} | "
                 f"{pct(s.loc['ridge_dl', 'rmse'])} | {pct(s.loc['ss_kalman', 'rmse'])} | **{pct(s.loc['ensemble', 'rmse'])}** | "
                 f"{s.loc['ensemble', 'r2_oos_vs_mean']:.2f} | {s.loc['ensemble', 'hit']:.2f} | {s.loc['ensemble', 'dm_p']:.3f} |")
    grp = summary.reset_index().merge(alt.meta[["group"]], left_on="code", right_index=True)
    g = grp.groupby(["group", "model"])["r2_oos_vs_mean"].mean().unstack("model")[["ar1", "ols", "ridge_dl", "ss_kalman", "ensemble"]]
    L.append("\n그룹 평균 R²_OOS (대 확장창 평균):\n")
    L.append("| 그룹 | AR(1) | OLS (F_q + AR) | A: Ridge DL | B: Kalman | Ensemble |")
    L.append("|---|---|---|---|---|---|")
    for gname, row in g.iterrows():
        L.append(f"| {gname} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
    L.append("\n![r2](figs/oos_r2.png)\n\n![paths](figs/oos_paths.png)\n")

    L.append("## 4. 모델 B 추정치 — 스무딩 θ 와 팩터 베타 (전체 표본)\n")
    L.append("| 시리즈 | θ | α(월) | σ_η(월) | " + " | ".join(f"β {f}" for f in factors) + " | " + " | ".join(f"(1−θ)β {f}" for f in factors) + " |")
    L.append("|---|---|---|---|" + "---|" * (2 * len(factors)))
    for c in order:
        r = pb.loc[c]
        L.append(f"| {c} | {r.theta:.2f} | {pct(r.alpha_m)} | {pct(r.sigma_eta_m)} | "
                 + " | ".join(f"{r[f'beta_{f}']:.2f}" for f in factors) + " | "
                 + " | ".join(f"{r[f'beta_reported_{f}']:.2f}" for f in factors) + " |")
    L.append("\nβ = 잠재(unsmoothed) 월간 로그수익률의 팩터 로딩, (1−θ)β = 보고 수익률에 나타나는 동분기 로딩. "
             "θ 가 클수록 보고 수익률이 과거를 끌고 온다 (부동산·VC 에서 크다).\n")
    L.append("![theta](figs/theta.png)\n\n![beta](figs/beta_economic.png)\n")

    L.append("## 5. 방법 요약\n")
    L.append("- **모델 A (Ridge 분포시차, reduced form)**: y_q = c + Σ_{l=0..4} β_l'F_{q−l} + φ y_{q−1} + e. 표준화 후 ridge, λ 는 훈련창 안 시간순 CV.")
    L.append("- **모델 B (월간 상태공간, structural)**: r*_m = α + β'F_m + η_m ; y_q = θ y_{q−1} + (1−θ) Σ_{m∈q} r*_m (Geltner 스무딩). 칼만 필터 MLE. 관측오차는 분기 자료만으로 σ_η 와 분리 식별되지 않아 0 으로 고정.")
    L.append("- **OLS 벤치마크**: y_q = c + φ y_{q−1} + b'F_q (동분기 팩터만). 완전 분기의 모델 B 점예측은 이 식의 제약형(b = (1−θ)β, 로그수익률)이라 OOS RMSE 가 거의 같다 — "
             "B 의 추가 가치는 부분 분기(QTD) 처리, θ/β 분해, 예측 표준편차다. A 의 lag 1~4 는 OOS 에서 도움이 되지 않는다(OLS 보다 약간 나쁨).")
    L.append("- **Ensemble** = A 와 B 의 단순평균.")
    L.append("- 팩터: 일간 → 월·분기 복리. 부분 기간(진행 중 분기, 2006Q1)은 QTD + (1−경과비율)×표본평균.")
    L.append("- 원지수가 같은 엑셀 라벨(Growth=VC, FoF=GP-Stake, 국내부동산=Core RE, 인프라 4종)은 한 시리즈로 추정하고 라벨로 되돌렸다.\n")
    L.append("## 6. 주의\n")
    L.append("- **의사 실시간**: Burgiss 류 지수는 실제로 1~2분기 늦게 발표되고 개정되지만, 여기서는 최종 빈티지를 쓰고 y_{q−1} 을 q 분기말에 안다고 가정했다. 실전 정확도는 이보다 낮다.")
    L.append("- 표본이 짧다 (OOS 34~57분기). DM 검정은 참고용.")
    L.append("- 부동산(Burgiss RE, MSCI GPFI, G-L 2)은 팩터 정보가 AR(1)을 이기지 못한다 — 감정평가 지수는 자기상관이 지배적이다.")
    L.append("- 팩터 4개만 쓴다. 부동산·인프라에는 실질금리·유동성·섹터 팩터를 더하면 개선 여지가 있다.")
    (rep / "nowcast_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"saved → {rep / 'nowcast_report.md'}  (+{len(list(figs.glob('*.png')))} figs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
