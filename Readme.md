#FFX: Fast Function Extraction

[![Build Status at Travis CI](https://travis-ci.org/natekupp/ffx.svg?branch=master)](https://travis-ci.org/natekupp/ffx)
[![Coverage Status](https://coveralls.io/repos/github/natekupp/ffx/badge.svg?branch=master)](https://coveralls.io/github/natekupp/ffx?branch=master)

FFX is a technique for symbolic regression. It is:

- __Fast__ - runtime 5-60 seconds, depending on problem size (1GHz cpu)
- __Scalable__ - 1000 input variables, no problem!
- __Deterministic__ - no need to "hope and pray".

## Installation
To install from PyPI, simply run:

	pip install ffx

## Usage
FFX can either be run in stand-alone mode, or within your existing Python code using its own API or a Scikit-learn style API. It installs both a command-line utility `runffx` and the Python module `ffx`.

__Standalone__

	runffx test train_X.csv train_y.csv test_X.csv test_y.csv

Use `runffx help` for more information on using the command-line utility.

__Python Module (run interface)__

The FFX Python module exposes a function, `ffx.run()`. The following snippet is a simple example of how to use FFX this way. Note that all arguments are expected to be of type `numpy.ndarray` or `pandas.DataFrame`.

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

__Scikit-Learn interface__

The FFX Python module also exposes a class, `ffx.FFXRegressor` which provides a Scikit-learn API, in particular `fit(X, y)`, `predict(X)`, and `score(X, y)` methods. In this API, all of the models produced by FFX (the whole Pareto front) are accessible after `fit()`ing as `_models`, but `predict()` and `score()` will use only the model of highest accuracy and highest complexity. Here is an example of usage.

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



## Dependencies
* python (tested on 2.5, 2.6, 2.7, 3.5, 3.6, 3.7)
* numpy (1.6.0+)
* scipy (0.9.0+)
* scikit-learn (1.5+)
* pandas (optional, enables support for labeled `pandas.DataFrame` datasets)


## Technical details
- Circuits-oriented description: [Slides](http://trent.st/content/2011-CICC-FFX-slides.ppt) [Paper](http://trent.st/content/2011-CICC-FFX-paper.pdf) (CICC 2011)
- AI-oriented description [Slides](http://trent.st/content/2011-GPTP-FFX-slides.pdf) [Paper](http://trent.st/content/2011-GPTP-FFX-paper.pdf) (GPTP 2011)


## References

1. McConaghy, FFX: Fast, Scalable, Deterministic Symbolic Regression Technology, _Genetic Programming Theory and Practice IX_, Edited by R. Riolo, E. Vladislavleva, and J. Moore, Springer, 2011.
2. McConaghy, High-Dimensional Statistical Modeling and Analysis of Custom Integrated Circuits, _Proc. Custom Integrated Circuits Conference_, Sept. 2011
