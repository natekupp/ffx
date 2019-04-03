# this is the tiny example in the README, and now it
# also prints out numBases and complexity for each model

import numpy as np
import ffx

train_X = np.array( [ (1.5,2,3), (4,5,6) ] ).T
train_y = np.array( [1,2,3])

test_X = np.array( [ (5.241,1.23, 3.125), (1.1,0.124,0.391) ] ).T
test_y = np.array( [3.03,0.9113,1.823])

models = ffx.run(train_X, train_y, test_X, test_y, ["a", "b"])
print("numBases: GP-complexity : model")
for model in models:
    yhat = model.simulate(test_X)
    print(model.numBases(), ":", model.complexity(), ": ", model)
