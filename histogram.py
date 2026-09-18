#!/usr/bin/env python3

import sys

from dslr import COLORS, COURSES, HOUSES, load_dataset, scores, show

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("histogram: this file needs matplotlib (pip install -r requirements.txt)")


def variance(values):
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def homogeneity(rows, course):
    """variance between house means / variance inside each house, low = homogeneous"""
    groups = [g for g in (scores(rows, course, h) for h in HOUSES) if g]
    if not groups:
        return float("inf")
    inside = sum(variance(g) for g in groups) / len(groups)
    between = variance([sum(g) / len(g) for g in groups])
    return between / inside if inside else float("inf")


def main():
    if len(sys.argv) != 2:
        print("usage: histogram.py dataset.csv", file=sys.stderr)
        return 1
    try:
        rows = load_dataset(sys.argv[1], COURSES, labeled=True)
    except (OSError, ValueError) as err:
        print(f"histogram: {err}", file=sys.stderr)
        return 1
    score = {c: homogeneity(rows, c) for c in COURSES}
    best = min(score, key=score.get)
    for c in sorted(COURSES, key=score.get):
        print(f"{c:<30} {score[c]:.4f}")
    print(f"most homogeneous: {best}")

    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for ax, course in zip(axes.flat, COURSES):
        for h in HOUSES:
            ax.hist(scores(rows, course, h), bins=20, alpha=0.5, density=True,
                    color=COLORS[h], label=h)
        ax.set_title(f"{course} ({score[course]:.3f})", fontsize=9,
                     fontweight="bold" if course == best else "normal")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(COURSES):]:
        ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower right")
    fig.suptitle(f"scores by house, most homogeneous: {best}", fontweight="bold")
    fig.tight_layout()
    show(fig, "histogram.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
