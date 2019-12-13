#!/usr/bin/env python

import numpy as np
import ffx

# This creates a dataset of 2 predictors
X = np.random.random((20, 2))
y = 0.1 * X[:, 0] + 0.5 * X[:, 1]

train_X, test_X = X[:10], X[10:]
train_y, test_y = y[:10], y[10:]

FFX = ffx.FFXRegressor()
FFX.fit(train_X, train_y)

print("Choosing the single most accurate model:")
print("Model:", FFX.model_)
print("Prediction:", FFX.predict(test_X))
print("Score:", FFX.score(test_X, test_y))
print("Complexity:", FFX.complexity())

print("\nLooking at all models, one by one:")
for model in FFX.models_:
    print("\nModel:", model)
    print("Prediction:", model.predict(test_X))
    print("Score:", model.score(test_X, test_y))
    print("Complexity:", model.complexity())
    
