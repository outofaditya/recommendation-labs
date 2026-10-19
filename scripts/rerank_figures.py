from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from rerank import BASES, FOLDER, RERANKERS

FIGURES = Path("report") / "figures"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TIERS = {"tail": "#86b6ef", "mid": "#3987e5", "head": "#1c5cab"}
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


# share of position weighted exposure per movie tier with the tail on the baseline so its growth reads from zero
def exposure(groups, name):
    tiers = groups[(groups.dimension == "tier") & (groups.metric == "exposure")]
    figure, axes = plt.subplots(len(BASES), 1, figsize=(6.3, 3.1), sharex=True)
    for ax, base in zip(axes, BASES):
        frame = tiers[tiers.model == base].set_index(["reranker", "group"])
        bars = {"Original": frame.xs(next(iter(RERANKERS)))["before"]} | {NAMES[r]: frame.xs(r)["after"] for r in RERANKERS}
        for y, (label, share) in enumerate(bars.items()):
            left = 0
            for tier, color in TIERS.items():
                ax.barh(y, share[tier], left=left, height=0.62, color=color, edgecolor="white", linewidth=1, label=tier if y == 0 else None)
                left += share[tier]
            ax.text(1.01, y, f"tail {100 * share['tail']:.1f}%", va="center", ha="left", fontsize=7.5, color=SECONDARY)
        ax.set_yticks(range(len(bars)), list(bars))
        ax.set_title(base, loc="left")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
    axes[-1].xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    axes[-1].set_xlabel("Share of position-weighted test exposure")
    handles = axes[0].get_legend_handles_labels()
    figure.legend(*handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), title=None)
    figure.tight_layout(rect=(0, 0, 0.93, 0.96))
    save(figure, name)


if __name__ == "__main__":
    tradeoff(pd.read_csv(FOLDER / "curves.csv"), "model", {base: base for base in BASES}, "rerank-tradeoff")
    pipelines = {"Members": "Re-rank members, then fuse", "Hybrid": "Fuse, then re-rank"}
    tradeoff(pd.read_csv(FOLDER / "hybrid" / "curves.csv"), "pipeline", pipelines, "rerank-hybrid")
    exposure(pd.read_csv(FOLDER / "groups.csv"), "rerank-exposure")
    print("figures in", FIGURES)
