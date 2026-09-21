#!/usr/bin/env python3

import sys

from dslr import COLORS, COURSES, HOUSES, load_dataset, pairs, scores, show

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("pair_plot: this file needs matplotlib (pip install -r requirements.txt)")

SHORT = {
    "Arithmancy": "Arith", "Astronomy": "Astro", "Herbology": "Herb",
    "Defense Against the Dark Arts": "DADA", "Divination": "Divin",
    "Muggle Studies": "Muggle", "Ancient Runes": "Runes", "History of Magic": "History",
    "Transfiguration": "Transf", "Potions": "Potions", "Care of Magical Creatures": "CoMC",
    "Charms": "Charms", "Flying": "Flying",
}


def main():
    if len(sys.argv) != 2:
        print("usage: pair_plot.py dataset.csv", file=sys.stderr)
        return 1
    try:
        rows = load_dataset(sys.argv[1], COURSES, labeled=True)
    except (OSError, ValueError) as err:
        print(f"pair_plot: {err}", file=sys.stderr)
        return 1
    n = len(COURSES)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.93)
    for i, y in enumerate(COURSES):
        for j, x in enumerate(COURSES):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == j:
                for h in HOUSES:
                    ax.hist(scores(rows, x, h), bins=15, alpha=0.5, density=True, color=COLORS[h])
            else:
                points = pairs(rows, x, y)
                for h in HOUSES:
                    ax.scatter([p[0] for p in points if p[2] == h], [p[1] for p in points if p[2] == h],
                               s=2, alpha=0.25, color=COLORS[h], label=h)
            if j == 0:
                ax.set_ylabel(SHORT[y], fontsize=7, rotation=45, ha="right")
            if i == n - 1:
                ax.set_xlabel(SHORT[x], fontsize=7, rotation=45, ha="right")
    fig.legend(*axes[0][1].get_legend_handles_labels(), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.965), markerscale=5)
    fig.suptitle("pair plot, scores by house", fontweight="bold", y=0.99)
    print("good features: separated humps on the diagonal, separated clouds elsewhere")
    show(fig, "pair_plot.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
