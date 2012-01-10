import core

def run(train_X, train_y, test_X, test_y, varnames=None):
    return core.MultiFFXModelFactory().build(train_X, train_y, test_X, test_y, varnames)
