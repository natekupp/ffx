# this is the tiny example in the README, and now it
# also prints out num_bases and complexity for each model

import ffx
import numpy as np

EXPECTED = [
    (0, 1, '2.00'),
    (1, 5, '1.85 + 0.0302*b'),
    (2, 10, '1.76 + 0.0720*log10(b) + 0.0389*b'),
    (3, 19, '1.72 - 0.0796*max(0,2.60-a) + 0.0705*b - 0.0701*max(0,2.40-a)'),
    (
        4,
        26,
        '1.71 - 0.0875*max(0,2.60-a) - 0.0796*max(0,2.40-a) + 0.0748*b - 0.000532*max(0,5.73-b)',
    ),
    (
        5,
        32,
        '(1.72 - 0.0802*max(0,2.60-a) - 0.0751*max(0,2.40-a) + 0.0714*b) / (1.0 + 0.0142*max(0,2.20-a) - 0.000721*b)',
    ),
    (
        6,
        39,
        '(1.71 - 0.0883*max(0,2.60-a) - 0.0862*max(0,2.40-a) + 0.0763*b - 0.00449*max(0,5.73-b)) / (1.0 + 0.0260*max(0,2.20-a) - 0.00130*b)',
    ),
    (
        8,
        63,
        '(1.75 + 0.0319*b + 0.0318*b - 0.0106*max(0,2.60-a) - 0.00616*b * max(0,5.73-b) + 2.53e-5*b^2 + 2.22e-5*b^2) / (1.0 + 0.0586*max(0,2.60-a) * max(0,2.40-a) + 0.0138*max(0,5.73-b) * max(0,2.60-a))',
    ),
]


def test_readme_example():
    train_X = np.array([(1.5, 2, 3), (4, 5, 6)]).T
    train_y = np.array([1, 2, 3])

    test_X = np.array([(5.241, 1.23, 3.125), (1.1, 0.124, 0.391)]).T
    test_y = np.array([3.03, 0.9113, 1.823])

    np.random.seed(0)

    models = ffx.run(train_X, train_y, test_X, test_y, ["a", "b"])
    assert [(model.numBases(), model.complexity(), str(model)) for model in models] == EXPECTED
