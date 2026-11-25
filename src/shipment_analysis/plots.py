"""Static charts for the README and report (matplotlib, PNG).

Colour rules used throughout:
* Identity lives on the axis labels, so single-series bars use one blue.
* Sentiment is polarity, so it uses a diverging red <-> grey <-> blue scale
  (red = negative, blue = positive). Never a rainbow.
* Line charts use the first categorical slots in a fixed order.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

BLUE = "#257de0"
RED = "#ec4647"
GREY = "#c9c9c9"
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]
DIVERGING = LinearSegmentedColormap.from_list("neg_pos", ["#c0302f", RED, "#efefef", "#519ced", "#1b64be"])
SENT_COLOURS = {"negative": RED, "neutral": GREY, "positive": BLUE}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2, "ytick.color": TEXT, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": TEXT, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9, va="bottom")


def courier_net_sentiment(summary: pd.DataFrame, path: Path):
    """summary: courier, n_reviews, net_sentiment"""
    _style()
    s = summary.sort_values("net_sentiment")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(s) + 1.4))
    colours = [BLUE if v >= 0 else RED for v in s["net_sentiment"]]
    ax.barh(s["courier"], s["net_sentiment"], color=colours, height=0.6)
    ax.axvline(0, color=TEXT_2, lw=1)
    lim = max(10, s["net_sentiment"].abs().max() * 1.25)
    ax.set_xlim(-lim, lim)
    for y, (v, n) in enumerate(zip(s["net_sentiment"], s["n_reviews"])):
        ax.text(v + (2 if v >= 0 else -2), y, f"{v:+.0f}  (n={n:,})", va="center",
                ha="left" if v >= 0 else "right", color=TEXT, fontsize=9)
    ax.set_xlabel("Net sentiment  (% positive minus % negative)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Net sentiment by courier", pad=22)
    _subtitle(ax, "Reviews that mention each courier")
    fig.savefig(path)
    plt.close(fig)


def aspect_heatmap(table: pd.DataFrame, counts: pd.DataFrame, path: Path, min_n: int = 5):
    """table/counts: index = courier, columns = aspect."""
    _style()
    masked = table.where(counts >= min_n)
    fig, ax = plt.subplots(figsize=(1.15 * masked.shape[1] + 2.5, 0.55 * masked.shape[0] + 1.8))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(masked.values.astype(float), cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(masked.shape[1]), [c.replace(" / ", " /\n").replace(" & ", " &\n") for c in masked.columns],
                  fontsize=8.5)
    ax.set_yticks(range(masked.shape[0]), masked.index)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, masked.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, masked.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            v = masked.values[i, j]
            if pd.isna(v):
                ax.text(j, i, "–", ha="center", va="center", color=TEXT_2, fontsize=8)
            else:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(v) > 0.55 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Mean sentence sentiment (−1 to +1)", color=TEXT_2)
    cb.outline.set_visible(False)
    ax.set_title("What each courier gets praised and blamed for", pad=22)
    _subtitle(ax, f"Red = negative, blue = positive. '–' = fewer than {min_n} mentions")
    fig.savefig(path)
    plt.close(fig)


def aspect_share_negative(share: pd.Series, path: Path):
    """share: index = aspect, values = share of negative reviews mentioning it."""
    _style()
    s = share.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1.4))
    ax.barh(s.index, s.values * 100, color=BLUE, height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v * 100 + 0.8, y, f"{v:.0%}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, max(10, s.max() * 100 * 1.15))
    ax.set_xlabel("% of negative reviews mentioning the aspect")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("What negative reviews complain about", pad=22)
    _subtitle(ax, "A review can mention more than one aspect")
    fig.savefig(path)
    plt.close(fig)


def model_comparison(scores: pd.DataFrame, path: Path):
    _style()
    s = scores.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(s) + 1.4))
    ax.barh(s["model"], s["macro_f1"], color=BLUE, height=0.6)
    for y, (f1, acc) in enumerate(zip(s["macro_f1"], s["accuracy"])):
        ax.text(f1 + 0.01, y, f"F1 {f1:.2f}  ·  acc {acc:.0%}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Macro F1 against star-rating labels")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Which sentiment model agrees best with the stars?", pad=22)
    _subtitle(ax, f"Stars 1–2 = negative, 3 = neutral, 4–5 = positive  ·  n = {int(s['n'].iloc[0]):,}")
    fig.savefig(path)
    plt.close(fig)


def label_share_by_stars(df: pd.DataFrame, label_col: str, model_name: str, path: Path):
    _style()
    rated = df.dropna(subset=["rating"])
    rated = rated.assign(stars=rated["rating"].round().clip(1, 5).astype(int))
    share = pd.crosstab(rated["stars"], rated[label_col], normalize="index").reindex(
        columns=["negative", "neutral", "positive"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(share))
    for lab in share.columns:
        vals = share[lab].values
        ax.barh([f"{s}★" for s in share.index], vals, left=left, color=SENT_COLOURS[lab], height=0.62,
                edgecolor="white", linewidth=2, label=lab.capitalize())
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if lab != "neutral" else TEXT)
        left += vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.24), fontsize=9)
    ax.set_title(f"{model_name} label vs the reviewer's star rating", pad=22)
    _subtitle(ax, "A good model is mostly red at 1★ and mostly blue at 5★")
    fig.savefig(path)
    plt.close(fig)


def monthly_trend(monthly: pd.DataFrame, path: Path):
    """monthly: month (Timestamp), courier, net_sentiment, n"""
    _style()
    couriers = monthly.groupby("courier")["n"].sum().sort_values(ascending=False).index[:4]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for colour, c in zip(SERIES, couriers):
        d = monthly[monthly["courier"] == c].sort_values("month")
        ax.plot(d["month"], d["net_sentiment"], color=colour, lw=2, marker="o", ms=4, label=c)
        last = d.iloc[-1]
        ax.text(last["month"], last["net_sentiment"], f"  {c}", color=TEXT, fontsize=9, va="center")
    ax.axhline(0, color=TEXT_2, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Net sentiment")
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, -0.3), fontsize=9)
    ax.set_title("Net sentiment over time", pad=22)
    _subtitle(ax, "Top 4 couriers by review volume, months with at least 10 mentions")
    fig.savefig(path)
    plt.close(fig)
