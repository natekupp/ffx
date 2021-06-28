# FFX: Fast Function Extraction

[![Build Status at Travis CI](https://travis-ci.org/natekupp/ffx.svg?branch=master)](https://travis-ci.org/natekupp/ffx)
[![Coverage Status](https://coveralls.io/repos/github/natekupp/ffx/badge.svg?branch=master)](https://coveralls.io/github/natekupp/ffx?branch=master)

FFX is a technique for symbolic regression. It is:

- **Fast** - runtime 5-60 seconds, depending on problem size
- **Scalable** - 1000 input variables, no problem!
- **Deterministic** - no need to "hope and pray".

## Installation

To install from PyPI, simply run:

```shell
pip install ffx
```

## Usage

FFX can either be run in stand-alone mode, or within your existing Python code using its own API or a Scikit-learn style API. It installs both a command-line utility `ffx` and the Python module `ffx`.

**Standalone**

```shell
ffx test train_X.csv train_y.csv test_X.csv test_y.csv
```

Use `ffx help` for more information on using the command-line utility.

**Python Module (run interface)**

The FFX Python module exposes a function, `ffx.run()`. The following snippet is a simple example of how to use FFX this way. Note that all arguments are expected to be of type `numpy.ndarray` or `pandas.DataFrame`.

```python
import numpy as np
import ffx

train_X = np.array( [ (1.5,2,3), (4,5,6) ] ).T
train_y = np.array( [1,2,3])

test_X = np.array( [ (5.241,1.23, 3.125), (1.1,0.124,0.391) ] ).T
test_y = np.array( [3.03,0.9113,1.823])

models = ffx.run(train_X, train_y, test_X, test_y, ["predictor_a", "predictor_b"])
for model in models:
    yhat = model.simulate(test_X)
    print(model)
```

**Scikit-Learn interface**

The FFX Python module also exposes a class, `ffx.FFXRegressor` which provides a Scikit-learn API, in particular `fit(X, y)`, `predict(X)`, and `score(X, y)` methods. In this API, all of the models produced by FFX (the whole Pareto front) are accessible after `fit()`ing as `_models`, but `predict()` and `score()` will use only the model of highest accuracy and highest complexity. Here is an example of usage.

```python
import numpy as np
import ffx

# This creates a dataset of 2 predictors
X = np.random.random((20, 2))
y = 0.1 * X[:, 0] + 0.5 * X[:, 1]

train_X, test_X = X[:10], X[10:]
train_y, test_y = y[:10], y[10:]

FFX = ffx.FFXRegressor()
FFX.fit(train_X, train_y)
print("Prediction:", FFX.predict(test_X))
print("Score:", FFX.score(test_X, test_y))
```

## Technical details

- Circuits-oriented description: [Slides](http://trent.st/content/2011-CICC-FFX-slides.ppt) [Paper](http://trent.st/content/2011-CICC-FFX-paper.pdf) (CICC 2011)
- AI-oriented description [Slides](http://trent.st/content/2011-GPTP-FFX-slides.pdf) [Paper](http://trent.st/content/2011-GPTP-FFX-paper.pdf) (GPTP 2011)

## References

1. McConaghy, FFX: Fast, Scalable, Deterministic Symbolic Regression Technology, _Genetic Programming Theory and Practice IX_, Edited by R. Riolo, E. Vladislavleva, and J. Moore, Springer, 2011.
2. McConaghy, High-Dimensional Statistical Modeling and Analysis of Custom Integrated Circuits, _Proc. Custom Integrated Circuits Conference_, Sept. 2011
