"""EDA 그림. 기준(보고/언스무딩)에 색이 따라간다: 보고 = slot1 blue, 언스무딩 = slot2 orange."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .eda import FAMILY_EN, REGIME_EN, REGIME_ORDER
from .plots import PAL, _save

BASE_COLOR = {"reported": PAL["s1"], "unsmoothed": PAL["s2"]}
BASE_NAME = {"reported": "reported (smoothed)", "unsmoothed": "unsmoothed"}
DIV = LinearSegmentedColormap.from_list("div", [PAL["s1"], PAL["neutral"], PAL["red"]])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])


def plot_stats_dumbbell(stats: pd.DataFrame, order: list[str], path: Path, cols=("ann_vol", "sharpe", "max_dd")):
    titles = {"ann_vol": "annualised vol", "sharpe": "Sharpe (vs 3M T-bill)", "max_dd": "max drawdown", "ann_mean": "annualised mean"}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 0.36 * len(order) + 1.4), sharey=True)
    ys = np.arange(len(order))[::-1]
    for ax, col in zip(axes, cols):
        a = stats.loc["reported"].reindex(order)[col].values
        b = stats.loc["unsmoothed"].reindex(order)[col].values
        for y, x0, x1 in zip(ys, a, b):
            ax.plot([x0, x1], [y, y], color=PAL["grid"], linewidth=2, zorder=1)
        ax.scatter(a, ys, s=40, color=BASE_COLOR["reported"], label=BASE_NAME["reported"], zorder=3, edgecolor=PAL["surface"])
        ax.scatter(b, ys, s=40, color=BASE_COLOR["unsmoothed"], label=BASE_NAME["unsmoothed"], zorder=3, edgecolor=PAL["surface"], marker="s")
        ax.set_title(titles.get(col, col), loc="left")
        if col in ("ann_vol", "max_dd", "ann_mean"):
            ax.xaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(order)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2)
    fig.tight_layout()
    _save(fig, path)


def plot_regime_heatmaps(rs: pd.DataFrame, family: str, order: list[str], path: Path):
    """4 패널: 연율 평균(보고, 언스무딩) · 연율 변동성(보고, 언스무딩)."""
    states = REGIME_ORDER[family]
    labels = [REGIME_EN.get(s, s) for s in states]
    panels = [("ann_mean", "reported"), ("ann_mean", "unsmoothed"), ("ann_vol", "reported"), ("ann_vol", "unsmoothed")]
    fig, axes = plt.subplots(1, 4, figsize=(1.0 * len(states) * 4 + 6, 0.36 * len(order) + 1.6), sharey=True)
    for ax, (metric, base) in zip(axes, panels):
        M = np.full((len(order), len(states)), np.nan)
        N = np.zeros_like(M)
        for i, c in enumerate(order):
            for j, s in enumerate(states):
                if (base, c, s) in rs.index:
                    M[i, j] = rs.loc[(base, c, s)].get(metric, np.nan); N[i, j] = rs.loc[(base, c, s)]["n"]
        if metric == "ann_mean":
            v = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
        else:
            im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=np.nanmax(M), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center", fontsize=7.5, color=PAL["ink"])
        ax.set_xticks(range(len(states))); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{'ann. mean' if metric == 'ann_mean' else 'ann. vol'} (%) — {BASE_NAME[base]}", loc="left", fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
    fig.suptitle(f"{FAMILY_EN.get(family, family)}  — cell = annualised %, quarters in that state", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_beta_dumbbell(betas: pd.DataFrame, order: list[str], factors: list[str], path: Path):
    fig, axes = plt.subplots(1, len(factors), figsize=(3.3 * len(factors), 0.36 * len(order) + 1.4), sharey=True)
    ys = np.arange(len(order))[::-1]
    for ax, f in zip(axes, factors):
        a = [betas.loc[("reported", c, "전체")][f"b_{f}"] if ("reported", c, "전체") in betas.index else np.nan for c in order]
        b = [betas.loc[("unsmoothed", c, "전체")][f"b_{f}"] if ("unsmoothed", c, "전체") in betas.index else np.nan for c in order]
        for y, x0, x1 in zip(ys, a, b):
            ax.plot([x0, x1], [y, y], color=PAL["grid"], linewidth=2, zorder=1)
        ax.scatter(a, ys, s=40, color=BASE_COLOR["reported"], zorder=3, edgecolor=PAL["surface"], label=BASE_NAME["reported"])
        ax.scatter(b, ys, s=40, color=BASE_COLOR["unsmoothed"], zorder=3, edgecolor=PAL["surface"], marker="s", label=BASE_NAME["unsmoothed"])
        ax.axvline(0, color=PAL["ink2"], linewidth=0.8)
        ax.set_title(f"beta to {f}", loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(order)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2)
    fig.tight_layout()
    _save(fig, path)


def plot_corr_pair(R: dict[str, pd.DataFrame], order: list[str], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, base in zip(axes, ["reported", "unsmoothed"]):
        C = R[base][order].corr(min_periods=20).values
        im = ax.imshow(C, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
        for i in range(len(order)):
            for j in range(len(order)):
                ax.text(j, i, f"{C[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.5, color=PAL["ink"])
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        ax.set_title(f"quarterly correlation — {BASE_NAME[base]}", loc="left"); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def plot_small_multiples(R: dict[str, pd.DataFrame], order: list[str], path: Path, kind: str, rq: pd.Series | None = None, ncols: int = 3):
    """kind = 'level' (누적지수) | 'rollvol' (12분기 롤링 연율 변동성). rq = 음영용 침체 라벨(분기)."""
    from .eda import rolling_vol
    n = len(order); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.1 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, order):
        for base in ["reported", "unsmoothed"]:
            r = R[base][c].dropna()
            y = (1 + r).cumprod() * 100 if kind == "level" else rolling_vol(r.to_frame(), 12).iloc[:, 0]
            ax.plot(y.index, y.values, color=BASE_COLOR[base], linewidth=1.5, label=BASE_NAME[base])
        if rq is not None:
            rec = rq[rq == "침체"].index
            for t in rec:
                ax.axvspan(t - pd.offsets.QuarterEnd(1), t, color=PAL["grid"], alpha=0.6, linewidth=0)
        ax.set_title(c, loc="left")
        if kind == "level":
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
    for ax in axes[n:]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("cumulative index (log, base 100)" if kind == "level" else "12-quarter rolling annualised vol", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


FACTOR_COLOR = {"growth": PAL["s1"], "term": PAL["s2"], "inflation": PAL["s3"], "credit": PAL["s4"]}


def plot_rolling_betas(rb: pd.DataFrame, order: list[str], factors: list[str], path: Path, title: str,
                       rq: pd.Series | None = None, ncols: int = 3):
    """시리즈별 패널, 팩터별 선 (색 고정). rb = rolling_betas() 결과."""
    n = len(order); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, order):
        if c not in rb.columns.get_level_values(0):
            ax.axis("off"); continue
        for f in factors:
            s = rb[(c, f)].dropna()
            ax.plot(s.index, s.values, color=FACTOR_COLOR[f], linewidth=1.5, label=f)
        ax.axhline(0, color=PAL["ink2"], linewidth=0.8)
        if rq is not None:
            for t in rq[rq == "침체"].index:
                ax.axvspan(t - pd.offsets.QuarterEnd(1), t, color=PAL["grid"], alpha=0.6, linewidth=0)
        ax.set_title(c, loc="left")
    for ax in axes[n:]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(factors), bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_rolling_beta_compare(rb_a: pd.DataFrame, rb_b: pd.DataFrame, order: list[str], factor: str, path: Path,
                              rq: pd.Series | None = None, ncols: int = 3):
    """한 팩터에 대한 롤링 베타, 보고 vs 언스무딩 비교."""
    n = len(order); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, order):
        for rb, base in [(rb_a, "reported"), (rb_b, "unsmoothed")]:
            if (c, factor) in rb.columns:
                s = rb[(c, factor)].dropna()
                ax.plot(s.index, s.values, color=BASE_COLOR[base], linewidth=1.5, label=BASE_NAME[base])
        ax.axhline(0, color=PAL["ink2"], linewidth=0.8)
        if rq is not None:
            for t in rq[rq == "침체"].index:
                ax.axvspan(t - pd.offsets.QuarterEnd(1), t, color=PAL["grid"], alpha=0.6, linewidth=0)
        ax.set_title(c, loc="left")
    for ax in axes[n:]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"20-quarter rolling beta to {factor}: reported vs unsmoothed", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)
