#!/usr/bin/env python3

import json
import sys

from dslr import HOUSES, house, load_dataset, number

try:
    import numpy as np
except ImportError:
    sys.exit("logreg_train: this file needs numpy (pip install -r requirements.txt)")

WEIGHTS_FILE = "weights.json"
SEED = 42
# astronomy is exactly -100 * defense against the dark arts (scatter_plot), and
# arithmancy / care of magical creatures look the same in every house (histogram)
FEATURES = [
    "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
    "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Charms", "Flying",
]
# learning rate, epochs, samples per update (None = all of them)
OPTIMIZERS = {"batch": (0.5, 2000, None), "minibatch": (0.1, 100, 32), "sgd": (0.01, 20, 1)}


def features(rows, names):
    """(m, n) matrix, nan where a score is missing"""
    return np.array([[number(r, f) for f in names] for r in rows], dtype=float)


def prepare(X, mean, std):
    """missing scores -> mean, standardize, add the bias column"""
    X = np.where(np.isnan(X), mean, X)
    return np.hstack([np.ones((len(X), 1)), (X - mean) / std])


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def loss(X, y, theta):
    """cross-entropy J(theta)"""
    h = np.clip(sigmoid(X @ theta), 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


def gradient_descent(X, y, lr, epochs, batch, rng):
    """theta -= lr * (1/m) * sum((h - y) * x), m = samples per update"""
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = order[i:i + batch]
            theta -= lr * X[idx].T @ (sigmoid(X[idx] @ theta) - y[idx]) / len(idx)
    return theta


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[2:] and sys.argv[2] not in OPTIMIZERS:
        print(f"usage: logreg_train.py dataset.csv [{'|'.join(OPTIMIZERS)}]", file=sys.stderr)
        return 1
    optimizer = sys.argv[2] if len(sys.argv) == 3 else "batch"
    try:
        rows = load_dataset(sys.argv[1], FEATURES, labeled=True)
    except (OSError, ValueError) as err:
        print(f"logreg_train: {err}", file=sys.stderr)
        return 1
    raw = features(rows, FEATURES)
    if np.isnan(raw).all(axis=0).any():
        print("logreg_train: a feature has no values", file=sys.stderr)
        return 1
    mean, std = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
    std[std == 0] = 1
    X = prepare(raw, mean, std)

    lr, epochs, batch = OPTIMIZERS[optimizer]
    batch = batch or len(X)
    rng = np.random.default_rng(SEED)
    print(f"{optimizer} gradient descent: lr {lr}, {epochs} epochs, batch size {batch}")
    thetas = {}
    for h in HOUSES:
        y = np.array([house(r) == h for r in rows], dtype=float)
        thetas[h] = gradient_descent(X, y, lr, epochs, batch, rng)
        print(f"{h:<11} loss {loss(X, y, thetas[h]):.4f}")
    predicted = np.argmax([sigmoid(X @ thetas[h]) for h in HOUSES], axis=0)
    truth = np.array([HOUSES.index(house(r)) for r in rows])
    print(f"training accuracy: {np.mean(predicted == truth):.4f}")

    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
             "weights": {h: t.tolist() for h, t in thetas.items()}}
    try:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(model, f, indent=2)
            f.write("\n")
    except OSError as err:
        print(f"logreg_train: {err}", file=sys.stderr)
        return 1
    print(f"weights saved to {WEIGHTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
