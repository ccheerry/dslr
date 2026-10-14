# dslr

dslr (Data Science x Logistic Regression) sorts students of Hogwarts into
their house from their marks in thirteen courses. A training set of 1 600
students with their house is described, plotted to choose the features and
used to fit a one-vs-all logistic regression; the fitted weights then sort the
400 students of the test set.

Everything is written in Python 3. `numpy` handles the matrix products of the
model and `matplotlib` the plots; the statistics of `describe.py` are computed
by hand with the standard library only.

## How it works

| Program | Role |
|---|---|
| `describe.py` | Count, mean, std, min, quartiles and max of every numeric column, with plain loops and no statistics function |
| `histogram.py` | One histogram per course, the four houses overlaid; `Arithmancy` is the most homogeneous course |
| `scatter_plot.py` | The two most similar courses: `Astronomy` and `Defense Against the Dark Arts`, with `r = -1` |
| `pair_plot.py` | Every course against every other, which shows the courses that separate the houses |
| `logreg_train.py` | Four one-vs-all classifiers on ten courses, saved to `weights.json` |
| `logreg_predict.py` | `weights.json` applied to a dataset, the result written to `houses.csv` |

`Arithmancy` and `Care of Magical Creatures` are left out because they look the
same in every house, and `Astronomy` because it is an exact multiple of
`Defense Against the Dark Arts`. A missing mark is replaced by the mean of its
course, and every course is standardized with the mean and std of the training
set, both saved in `weights.json` so prediction applies the same transform.

The probability of each house is `sigmoid(theta . x)`, with `x` the
standardized marks after a 1 for the bias. Gradient descent minimizes the
cross-entropy cost with `theta -= lr * 1/m * X^T (h - y)`, and each student
goes to the house with the highest probability. The training accuracy is
98.19 %, and between 98.4 % and 98.8 % on the students left out of random
80 / 20 splits.

## Dependencies

| Dependency | Version | Debian / Ubuntu | Arch |
|---|---|---|---|
| Python 3 with `venv` | 3.8 or newer, tested with 3.14 | `python3 python3-venv` | `python` |
| GNU Make | any | `make` | `make` |
| `numpy`, `matplotlib` | `numpy >= 1.21`, `matplotlib >= 3.5` | installed into `.venv/` by `make install` | same |
| Tk | optional, for the plot windows | `python3-tk` | `tk` |

## Build and run

```
make install               # .venv/ with numpy and matplotlib
make run                   # describe, the three plots, training and prediction
make describe              # each program on its own: describe, histogram,
make pair_plot             #   scatter_plot, pair_plot, train, predict
make train OPTIMIZER=sgd   # batch (default), minibatch or sgd
make clean                 # weights.json, houses.csv and the PNGs
make fclean                # clean, and the .venv
```

`make` alone lists the targets. Nothing is installed outside `.venv/`. Each
plot opens in a window and the next step starts once it is closed; without a
display or without Tk, the figure is saved as a PNG in the repository root.
With the environment active, every program also runs on its own, for example
`python srcs/describe.py datasets/dataset_train.csv`.

A missing or malformed file, a dataset without the needed columns or without
labeled rows, and a corrupt `weights.json` print one line on standard error and
exit with status 1.

## Bonus

- `describe.py` adds five fields to the standard eight: `missing`, `var`,
  `range`, `iqr` and `skew`.
- `logreg_train.py` offers two optimizers besides `batch`: `minibatch` (32 rows
  per step) and `sgd` (one row per step). All three reach the same accuracy.

## Layout

```
dslr/
├── Makefile           # install, every program, clean
├── requirements.txt   # numpy and matplotlib
├── datasets/          # dataset_train.csv (with houses), dataset_test.csv (house column empty)
├── docs/              # documentation and the three plots rendered from the training set
└── srcs/              # the six programs and dslr.py, the module they share
```
