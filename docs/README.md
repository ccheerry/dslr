# dslr

Data Science x Logistic Regression. The project rebuilds the Sorting Hat of
Hogwarts from data: a training set of 1 600 students with their marks in
thirteen courses and their house, a test set of 400 students without the
house. The programs describe the data, plot it to choose the features, train
a logistic regression classifier and use it to sort the test set.

Everything is written in Python 3. `numpy` is used for the matrix products of
the model and `matplotlib` for the plots. The statistics of `describe.py` are
computed by hand and that script needs only the standard library.

## Layout

```
datasets/   dataset_train.csv (labeled), dataset_test.csv (house column empty)
docs/       this file and the three plots rendered from the training set
srcs/       the six programs and the module they share
```

All commands below are run from the root of the repository. The files a
program writes (`weights.json`, `houses.csv`, PNGs) land in the current
directory and are ignored by git.

## Install

```
pip install -r requirements.txt
```

## Programs

### describe.py

```
python3 srcs/describe.py datasets/dataset_train.csv
```

Prints one column per numeric feature and thirteen rows: `count`, `mean`,
`std`, `min`, `25%`, `50%`, `75%`, `max`, then the extra fields `missing`,
`var`, `range`, `iqr` and `skew`. Every value is computed with plain loops,
without `len`, `sum`, `min`, `max` or the `statistics` module. The variance is
the sample one (divided by `n - 1`) and the percentiles interpolate linearly
between ranks, so the output matches `pandas.DataFrame.describe`. The table
is split into blocks that fit the width of the terminal.

### histogram.py

```
python3 srcs/histogram.py datasets/dataset_train.csv
```

Answers the question "which course has a homogeneous score distribution
between the four houses". For every course it computes the variance of the
four house means divided by the mean variance inside the houses, prints the
courses sorted by that ratio and names the lowest, then draws one histogram
per course with the four houses overlaid. `Arithmancy` and `Care of Magical
Creatures` come out flat: their curves are the same in every house.

### scatter_plot.py

```
python3 srcs/scatter_plot.py datasets/dataset_train.csv
```

Answers "which two features are similar". It computes the Pearson correlation
of every pair of courses, prints the five strongest and plots the strongest
one. `Astronomy` and `Defense Against the Dark Arts` have `r = -1`: one is
exactly `-100` times the other on every student.

### pair_plot.py

```
python3 srcs/pair_plot.py datasets/dataset_train.csv
```

Draws the 13 x 13 scatter matrix of the courses, with histograms on the
diagonal, coloured by house. It is the picture used to decide which features
to feed the model: the ones whose histograms show separate humps and whose
scatter plots show separate clouds.

### logreg_train.py

```
python3 srcs/logreg_train.py datasets/dataset_train.csv [batch|minibatch|sgd]
```

Trains a one-vs-all logistic regression on ten courses (`Arithmancy`,
`Care of Magical Creatures` and `Astronomy` are left out, see above). Missing
marks are replaced by the mean of their column, every column is standardized
and a bias column is added. Four classifiers are fitted with gradient
descent, one per house, each on `y = 1` for that house and `y = 0` for the
others. The optimizer is `batch` by default; the other two are `minibatch`
(32 rows per step) and `sgd` (one row per step). The program prints the
final loss of each classifier and the training accuracy, then writes
`weights.json` with the feature list, the mean and std used to standardize,
and the weights of the four houses.

### logreg_predict.py

```
python3 srcs/logreg_predict.py datasets/dataset_test.csv weights.json
```

Loads the model, applies the same preprocessing with the mean and std stored
in it, computes the four probabilities of every student and keeps the
largest. Writes `houses.csv` with two columns, `Index` and `Hogwarts House`,
one line per student.

## The model

The probability that a student belongs to a house is `sigmoid(theta . x)`,
with `x` the standardized marks preceded by a 1 for the bias. The weights
`theta` minimize the cross-entropy cost

```
J(theta) = -1/m * sum( y * log(h) + (1 - y) * log(1 - h) )
```

by repeating the update `theta -= lr * 1/m * X^T (h - y)`, where `m` is the
number of rows used in the step: all of them for batch gradient descent, 32
for mini-batch, 1 for stochastic. Smaller batches use a smaller learning rate
and fewer epochs. Training takes about a second at most.

## Results

The three optimizers reach a training accuracy of 98.19 %. On random 80 / 20
splits of the training set, the accuracy on the students left out is between
98.4 % and 98.8 %. The statistics printed by `describe.py` agree with `numpy`
to machine precision.

## Errors

A missing or unreadable file, a file that is not a CSV, a dataset without
the needed columns or without any labeled row, a feature with no values and a
malformed `weights.json` all print a message on standard error and exit with
status 1. When matplotlib has no interactive backend, the plotting programs
save a PNG in the current directory instead of opening a window.

## Requirements

- Python 3 (tested with 3.14; nothing newer than 3.8 is used)
- `numpy >= 1.21`, `matplotlib >= 3.5`
