#!/usr/bin/env python3

import csv
import json
import sys

from dslr import load_dataset

try:
    import numpy as np
except ImportError:
    sys.exit("logreg_predict: this file needs numpy (pip install -r requirements.txt)")

from logreg_train import features, prepare, sigmoid

OUTPUT_FILE = "houses.csv"


def load_model(path):
    """weights.json written by logreg_train"""
    with open(path) as f:
        try:
            model = json.load(f)
            names = list(model["features"])
            mean = np.array(model["mean"], dtype=float)
            std = np.array(model["std"], dtype=float)
            houses = list(model["weights"])
            W = np.array([model["weights"][h] for h in houses], dtype=float)
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError(f"{path}: corrupt weights file ({err}), run logreg_train.py") from None
    if not mean.shape == std.shape == (len(names),) or W.shape[1:] != (len(names) + 1,):
        raise ValueError(f"{path}: corrupt weights file (wrong sizes), run logreg_train.py")
    return names, mean, std, houses, W


def main():
    if len(sys.argv) != 3:
        print("usage: logreg_predict.py dataset.csv weights.json", file=sys.stderr)
        return 1
    try:
        names, mean, std, houses, W = load_model(sys.argv[2])
        rows = load_dataset(sys.argv[1], names)
    except (OSError, ValueError) as err:
        print(f"logreg_predict: {err}", file=sys.stderr)
        return 1
    X = prepare(features(rows, names), mean, std)
    # one vs all: the house whose classifier is the most confident
    predicted = sigmoid(X @ W.T).argmax(axis=1)
    try:
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Index", "Hogwarts House"])
            for i, (row, p) in enumerate(zip(rows, predicted)):
                writer.writerow([row.get("Index", i), houses[p]])
    except OSError as err:
        print(f"logreg_predict: {err}", file=sys.stderr)
        return 1
    print(f"{len(rows)} predictions saved to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
