import ffx
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

EXPECTED_MODELS = [
    (0, 1, '0.298'),
    (1, 5, '0.102 + 0.395*X1'),
    (2, 9, '0.0141 + 0.485*X1 + 0.0861*X0'),
    (
        7,
        42,
        '0.0924 + 0.372*X1 - 0.0743*max(0,0.867-X1) + 0.0658*X0 + 0.0359*X0 * X1 + 0.0201*max(0,X1-0.200) + 0.00932*X1^2 - 0.00504*max(0,0.867-X0)',
    ),
]


def test_sklearn_api():
    np.random.seed(0)

    n_samples = 10000

    # This creates a dataset of 2 predictors
    X = np.random.random((n_samples, 2))  # pylint: disable=no-member
    y = 0.1 * X[:, 0] + 0.5 * X[:, 1]

    train_X, test_X = X[: int(n_samples / 2)], X[int(n_samples / 2) :]
    train_y, test_y = y[: int(n_samples / 2)], y[int(n_samples / 2) :]

    FFX = ffx.FFXRegressor()
    FFX.fit(train_X, train_y)

    # Best model
    assert (
        str(FFX.model_)
        == '0.0924 + 0.372*X1 - 0.0743*max(0,0.867-X1) + 0.0658*X0 + 0.0359*X0 * X1 + 0.0201*max(0,X1-0.200) + 0.00932*X1^2 - 0.00504*max(0,0.867-X0)'
    )
    assert FFX.model_.numBases() == 7
    assert FFX.score(test_X, test_y) == 0.9984036148094735
    assert FFX.complexity() == 42

    assert [
        (model.numBases(), model.complexity(), str(model)) for model in FFX.models_
    ] == EXPECTED_MODELS


def test_check_estimator():
    # Pass instance of estimator to run sklearn's built in estimator check
    check_estimator(ffx.FFXRegressor())

