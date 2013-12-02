#!/usr/bin/env python

import numpy as np
import ffx

# This creates a dataset of 1 predictor
train_X = np.array([[0, 1, 2, 3]]).T
train_y = np.array([0, 1, 4, 9])

test_X = np.array([[4, 5, 6, 7]]).T
test_y = np.array([16, 25, 36, 49])

models = ffx.run(train_X, train_y, test_X, test_y, ["x"])

print('True model: y = x^2')
print('Results:')
print('Num bases,Test error (%),Model\n')
for model in models:
    print('%10s, %13s, %s\n' %
          ('%d' % model.numBases(), '%.4f' % (model.test_nmse * 100.0), model))
