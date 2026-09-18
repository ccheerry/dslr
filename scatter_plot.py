#!/usr/bin/env python3

import itertools
import math
import sys

from dslr import COLORS, COURSES, HOUSES, load_dataset, pairs, show

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("scatter_plot: this file needs matplotlib (pip install -r requirements.txt)")


def pearson(points):
    """correlation, |r| = 1 means both features carry the same information"""
    n = len(points)
    if n < 2:
        return 0.0
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    cov = sum((x - mx) * (y - my) for x, y, _ in points)
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in points))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in points))
    return cov / (sx * sy) if sx and sy else 0.0


def main():
    if len(sys.argv) != 2:
        print("usage: scatter_plot.py dataset.csv", file=sys.stderr)
        return 1
    try:
        rows = load_dataset(sys.argv[1], COURSES, labeled=True)
    except (OSError, ValueError) as err:
        print(f"scatter_plot: {err}", file=sys.stderr)
        return 1
    r = {(a, b): pearson(pairs(rows, a, b)) for a, b in itertools.combinations(COURSES, 2)}
    ranking = sorted(r, key=lambda k: abs(r[k]), reverse=True)
    for a, b in ranking[:5]:
        print(f"{a:<30} {b:<30} r = {r[a, b]:.4f}")
    a, b = ranking[0]
    print(f"most similar: {a} and {b}")

    points = pairs(rows, a, b)
    fig, ax = plt.subplots(figsize=(8, 6))
    for h in HOUSES:
        ax.scatter([p[0] for p in points if p[2] == h], [p[1] for p in points if p[2] == h],
                   s=15, alpha=0.5, color=COLORS[h], label=h)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"most similar features (r = {r[a, b]:.4f})")
    ax.legend()
    fig.tight_layout()
    show(fig, "scatter_plot.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
