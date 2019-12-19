import ffx
import numpy as np

EXPECTED = [
    (0, 1, '3.50'),
    (1, 7, '0.640 + 0.817*x^2'),
    (2, 11, '0.0846 + 0.972*x^2 + 0.00984*x'),
    (
        6,
        31,
        '(0.0955 + 0.488*x^2 + 0.468*x^2 + 0.00638*x + 0.00124*x) / (1.0 - 0.00336*x - 0.00213*x)',
    ),
]


def test_x_square():
    np.random.seed(0)

    # This creates a dataset of 1 predictor
    train_X = np.array([[0, 1, 2, 3]]).T
    train_y = np.array([0, 1, 4, 9])

    test_X = np.array([[4, 5, 6, 7]]).T
    test_y = np.array([16, 25, 36, 49])

    models = ffx.run(train_X, train_y, test_X, test_y, ["x"])
    assert [(model.numBases(), model.complexity(), str(model)) for model in models] == EXPECTED
