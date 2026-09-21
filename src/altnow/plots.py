"""그림. 팔레트는 검증된 기본 카테고리 순서(blue, orange, aqua, yellow)를 고정 슬롯으로 쓴다 — 시리즈에 색이 따라간다."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

PAL = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e8e7e3",
       "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4", "neutral": "#f0efec", "red": "#e34948"}
MODEL_COLOR = {"ridge_dl": PAL["s1"], "ss_kalman": PAL["s2"], "ensemble": PAL["s3"], "ar1": PAL["s4"], "ols": PAL["s5"]}
MODEL_MARK = {"ridge_dl": "o", "ss_kalman": "s", "ensemble": "D", "ar1": "^", "ols": "v"}
MODEL_NAME = {"ridge_dl": "A: Ridge DL", "ss_kalman": "B: Kalman", "ensemble": "Ensemble", "ar1": "AR(1)", "ols": "OLS (F_q + AR)", "mean": "Mean"}

plt.rcParams.update({"figure.facecolor": PAL["surface"], "axes.facecolor": PAL["surface"], "axes.edgecolor": PAL["grid"],
                     "axes.labelcolor": PAL["ink2"], "xtick.color": PAL["ink2"], "ytick.color": PAL["ink2"],
                     "text.color": PAL["ink"], "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "savefig.dpi": 150})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_oos_r2(summary: pd.DataFrame, order: list[str], path: Path, metric: str = "r2_oos_vs_mean"):
    """시리즈 × 모델 R²_OOS 점그림 (벤치마크 = 확장창 평균)."""
    models = ["ar1", "ols", "ridge_dl", "ss_kalman", "ensemble"]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(order) + 1.2))
    ys = np.arange(len(order))[::-1]
    for m in models:
        v = [summary.loc[(c, m), metric] if (c, m) in summary.index else np.nan for c in order]
        ax.scatter(v, ys, s=42, marker=MODEL_MARK[m], color=MODEL_COLOR[m], edgecolor=PAL["surface"], linewidth=1.2,
                   label=MODEL_NAME[m], zorder=3)
    ax.axvline(0, color=PAL["ink2"], linewidth=0.8, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(order)
    ax.set_xlabel("out-of-sample R² vs. expanding mean (higher is better)")
    ax.set_xlim(left=max(-1.0, ax.get_xlim()[0]))
    ax.legend(loc="lower left", ncol=5, bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_oos_paths(preds: dict[str, pd.DataFrame], order: list[str], path: Path, ncols: int = 3):
    """실제 vs nowcast (ensemble, AR(1)) 소형 다중 패널."""
    n = len(order); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.1 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, code in zip(axes, order):
        df = preds[code]
        ax.plot(df.index, df["y"], color=PAL["ink"], linewidth=1.6, label="actual")
        ax.plot(df.index, df["ensemble"], color=PAL["s3"], linewidth=1.6, label=MODEL_NAME["ensemble"])
        ax.plot(df.index, df["ar1"], color=PAL["s4"], linewidth=1.2, label=MODEL_NAME["ar1"])
        ax.axhline(0, color=PAL["grid"], linewidth=0.8)
        ax.set_title(code, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes[n:]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, path)


def plot_latest_nowcast(now: pd.DataFrame, order: list[str], path: Path, quarters: list[pd.Timestamp]):
    """최신 nowcast (ensemble) — 시리즈별 분기 막대. 칼만 sd 를 얇은 오차선으로."""
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
    ys = np.arange(len(order))[::-1]
    k = len(quarters); h = 0.8 / k
    for j, q in enumerate(quarters):
        sub = now[now.quarter == q].set_index("code").reindex(order)
        col = [PAL["s1"], PAL["s2"], PAL["s3"], PAL["s4"]][j % 4]
        lab = f"{q.year}Q{(q.month - 1) // 3 + 1}" + (" (QTD)" if (sub["type"] == "partial(QTD)").any() else "")
        ax.barh(ys - 0.4 + h * (j + 0.5), sub["ensemble"].values, height=h - 0.06, color=col, label=lab, zorder=3)
        ax.errorbar(sub["ensemble"].values, ys - 0.4 + h * (j + 0.5), xerr=sub["ss_sd"].values, fmt="none",
                    ecolor=PAL["ink2"], elinewidth=0.7, capsize=0, zorder=4)
    ax.axvline(0, color=PAL["ink2"], linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(order)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("nowcast quarterly return (ensemble of A and B); whisker = Kalman 1 s.d.")
    ax.legend(loc="lower left", ncol=k, bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_theta(params_b: pd.DataFrame, order: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(order) + 1))
    ys = np.arange(len(order))[::-1]
    ax.barh(ys, params_b.loc[order, "theta"].values, height=0.62, color=PAL["s1"], zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels(order)
    ax.set_xlabel("Geltner smoothing θ (0 = no smoothing)")
    ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_beta_heatmap(params_b: pd.DataFrame, order: list[str], factors: list[str], path: Path, reported: bool = False):
    cols = [f"beta_reported_{f}" if reported else f"beta_{f}" for f in factors]
    M = params_b.loc[order, cols].values
    vmax = float(np.nanmax(np.abs(M)))
    cmap = LinearSegmentedColormap.from_list("div", [PAL["s1"], PAL["neutral"], PAL["red"]])
    fig, ax = plt.subplots(figsize=(1.1 * len(factors) + 2.2, 0.36 * len(order) + 1))
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect="auto")
    ax.set_xticks(range(len(factors))); ax.set_xticklabels(factors)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=PAL["ink"])
    ax.set_title(("reported (1−θ)β" if reported else "economic β (unsmoothed)") + " — monthly log-return loading", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    _save(fig, path)
