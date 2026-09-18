#!/usr/bin/env python3

import math
import shutil
import sys

from dslr import load_dataset

# every statistic is computed by hand, the subject forbids anything that
# does the job (count, mean, std, min, max, percentile...)
FIELDS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max",
          "missing", "var", "range", "iqr", "skew"]


def numeric_columns(rows):
    """columns where every non-empty cell is a number, nan/inf count as missing"""
    columns = {}
    for col in rows[0]:
        cells = [(r[col] or "").strip() for r in rows]
        try:
            values = [float(c) for c in cells if c]
        except ValueError:
            continue
        if values:
            columns[col] = [v for v in values if math.isfinite(v)]
    return columns


def count(values):
    n = 0
    for _ in values:
        n += 1
    return n


def total(values):
    s = 0.0
    for x in values:
        s += x
    return s


def mean(values):
    n = count(values)
    return total(values) / n if n else math.nan


def variance(values):
    """sample variance, divided by n - 1 like pandas"""
    n = count(values)
    if n < 2:
        return math.nan
    mu = mean(values)
    return total([(x - mu) ** 2 for x in values]) / (n - 1)


def minimum(values):
    m = math.nan
    for x in values:
        if m != m or x < m:
            m = x
    return m


def maximum(values):
    m = math.nan
    for x in values:
        if m != m or x > m:
            m = x
    return m


def percentile(ordered, p):
    """linear interpolation between the two closest ranks, like numpy/pandas"""
    n = count(ordered)
    if n == 0:
        return math.nan
    i = p / 100 * (n - 1)
    lo = int(i)
    if lo + 1 >= n:
        return ordered[lo]
    return ordered[lo] + (i - lo) * (ordered[lo + 1] - ordered[lo])


def skewness(values):
    """mean of ((x - mean) / std)^3, 0 = symmetric"""
    n = count(values)
    if n < 2:
        return math.nan
    mu = mean(values)
    sigma = math.sqrt(total([(x - mu) ** 2 for x in values]) / n)
    return total([((x - mu) / sigma) ** 3 for x in values]) / n if sigma else 0.0


def describe(values, rows):
    ordered = sorted(values)
    q1, q3 = percentile(ordered, 25), percentile(ordered, 75)
    lo, hi = minimum(values), maximum(values)
    return {
        "count": count(values), "mean": mean(values), "std": math.sqrt(variance(values)),
        "min": lo, "25%": q1, "50%": percentile(ordered, 50), "75%": q3, "max": hi,
        "missing": rows - count(values), "var": variance(values),
        "range": hi - lo, "iqr": q3 - q1, "skew": skewness(values),
    }


def print_table(stats):
    """columns split in blocks that fit the terminal, like pandas"""
    cells = {c: [f"{float(stats[c][f]):.6f}" for f in FIELDS] for c in stats}
    widths = {c: maximum([count(c)] + [count(v) for v in cells[c]]) + 2 for c in stats}
    screen = shutil.get_terminal_size((120, 24)).columns
    blocks, used = [[]], 8
    for c in stats:
        if blocks[-1] and used + widths[c] > screen:
            blocks.append([])
            used = 8
        blocks[-1].append(c)
        used += widths[c]
    for b, cols in enumerate(blocks):
        if b:
            print()
        print(" " * 8 + "".join(f"{c:>{widths[c]}}" for c in cols))
        for k, field in enumerate(FIELDS):
            print(f"{field:<8}" + "".join(f"{cells[c][k]:>{widths[c]}}" for c in cols))


def main():
    if len(sys.argv) != 2:
        print("usage: describe.py dataset.csv", file=sys.stderr)
        return 1
    try:
        rows = load_dataset(sys.argv[1])
    except (OSError, ValueError) as err:
        print(f"describe: {err}", file=sys.stderr)
        return 1
    columns = numeric_columns(rows)
    if not columns:
        print(f"describe: {sys.argv[1]}: no numerical feature", file=sys.stderr)
        return 1
    print_table({c: describe(values, count(rows)) for c, values in columns.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
