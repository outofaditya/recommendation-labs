from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from rerank import BASES, FOLDER, RERANKERS

FIGURES = Path("report") / "figures"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TIERS = {"tail": "#86b6ef", "mid": "#3987e5", "head": "#1c5cab"}
# taste groups are ordered so they take one orange ramp light to dark beside the blue movie tiers
TASTES = {"niche": "#f0956a", "diverse": "#e0602b", "blockbuster": "#a3401a"}
NAMES = {"MMR": "MMR", "Calibrated": "Steck", "xQuAD": "xQuAD", "CP": "CP"}
# each re-ranker's own target with lower-is-better axes reversed so improvement always points right
TARGETS = {
    "MMR": ("ild", "ILD (higher is better)", False),
    "Calibrated": ("miscalibration", "Genre miscalibration, KL (lower is better)", True),
    "xQuAD": ("tail", "Tail share of the list (higher is better)", False),
    "CP": ("upd", "UPD (lower is better)", True),
}
BUDGET = -5

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png")
    plt.close(figure)


# one panel per re-ranker with validation ndcg relative to no re-ranking so the 5% budget is one line for every series
def tradeoff(curves, by, labels, name):
    figure, axes = plt.subplots(2, 2, figsize=(6.3, 3.6))
    for ax, (reranker, (metric, label, lower)) in zip(axes.flat, TARGETS.items()):
        for color, (key, text) in zip(SERIES, labels.items()):
            line = curves[(curves[by] == key) & (curves.reranker == reranker)].sort_values("lambda")
            change = 100 * (line.ndcg / line.ndcg.iloc[0] - 1)
            ax.plot(line[metric], change, color=color, linewidth=1.5, solid_capstyle="round", label=text)
            chosen = line.chosen.to_numpy()
            ax.plot(line[metric][chosen], change[chosen], "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
        ax.axhline(BUDGET, color=MUTED, linewidth=0.8, zorder=1)
        ax.text(1, BUDGET, "5% budget", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=SECONDARY, fontsize=7)
        ax.set_ylim(-20, 1.5)
        ax.set_title(f"{NAMES[reranker]}", loc="left")
        ax.set_xlabel(label)
        if lower:
            ax.invert_xaxis()
    figure.supylabel("Validation NDCG@10 change (%)", fontsize=9)
    handles = axes[0, 0].get_legend_handles_labels()
    figure.legend(*handles, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save(figure, name)


# each re-ranker's mean change per group with its 95% interval, groups ordered light to dark
def dots(ax, rows, colors, label):
    for k, reranker in enumerate(RERANKERS):
        for j, (group, color) in enumerate(colors.items()):
            row = rows[(rows.reranker == reranker) & (rows.group == group)].iloc[0]
            spread = [[row.change - row.low], [row.high - row.change]]
            ax.errorbar(
                row.change,
                k + (j - 1) * 0.22,
                xerr=spread,
                fmt="o",
                color=color,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=1.2,
                elinewidth=1.3,
                capsize=0,
                label=group if k == 0 else None,
            )
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(RERANKERS)), [NAMES[r] for r in RERANKERS])
    ax.set_ylim(len(RERANKERS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(label)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8)


# who pays per user group beside whose relevant movies are still found per movie group so one figure covers both sides of 3.4
def impact(groups, base, name):
    figure, (users, movies) = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    rows = groups[groups.model == base]
    dots(users, rows[(rows.dimension == "taste within activity") & (rows.metric == "ndcg")], TASTES, "NDCG@10 change per user")
    dots(movies, rows[(rows.dimension == "tier") & (rows.metric == "recall")], TIERS, "Recall@10 change on the tier's test movies")
    movies.tick_params(axis="y", labelleft=False)
    users.set_title(f"Users ({base}): who pays", loc="left")
    movies.set_title(f"Movies ({base}): whose relevant films are found", loc="left")
    figure.tight_layout(w_pad=2.5)
    save(figure, name)


if __name__ == "__main__":
    tradeoff(pd.read_csv(FOLDER / "curves.csv"), "model", {base: base for base in BASES}, "rerank-tradeoff")
    pipelines = {"Members": "Re-rank members, then fuse", "Hybrid": "Fuse, then re-rank"}
    tradeoff(pd.read_csv(FOLDER / "hybrid" / "curves.csv"), "pipeline", pipelines, "rerank-hybrid")
    impact(pd.read_csv(FOLDER / "groups.csv"), "EASE", "rerank-impact")
    print("figures in", FIGURES)
