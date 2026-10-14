PYTHON		?= python3
VENV		= .venv
PY			= $(VENV)/bin/python
STAMP		= $(VENV)/.installed

TRAIN		= datasets/dataset_train.csv
TEST		= datasets/dataset_test.csv
OPTIMIZER	?= batch

PLOTS		= histogram scatter_plot pair_plot


help:
	@echo "make install           .venv with numpy and matplotlib"
	@echo "make run               describe, the three plots, training and prediction"
	@echo "make describe          statistics of the training set"
	@echo "make histogram         homogeneity of every course across the houses"
	@echo "make scatter_plot      the two most similar courses"
	@echo "make pair_plot         every course against every other"
	@echo "make train             logistic regression, weights.json (OPTIMIZER=batch|minibatch|sgd)"
	@echo "make predict           houses.csv for the test set"
	@echo "make clean             weights.json, houses.csv and the PNGs"
	@echo "make fclean            clean, and the .venv"

run: describe plots train predict

install: $(STAMP)

$(STAMP): requirements.txt
	@command -v $(PYTHON) >/dev/null || { echo "$(PYTHON) not found: install Python 3.8 or newer" >&2; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q -r requirements.txt
	@touch $@

describe: $(STAMP)
	$(PY) srcs/describe.py $(TRAIN)

$(PLOTS): $(STAMP)
	$(PY) srcs/$@.py $(TRAIN)

plots: $(PLOTS)

train: $(STAMP)
	$(PY) srcs/logreg_train.py $(TRAIN) $(OPTIMIZER)

predict: $(STAMP)
	@[ -f weights.json ] || $(MAKE) --no-print-directory train
	$(PY) srcs/logreg_predict.py $(TEST) weights.json

clean:
	rm -f weights.json houses.csv $(PLOTS:%=%.png)
	rm -rf srcs/__pycache__

fclean: clean
	rm -rf $(VENV)

re: fclean install

.PHONY: help run install describe $(PLOTS) plots train predict clean fclean re
