import ffx
import numpy as np
from ffx.core import (
    INF,
    OP_ABS,
    OP_GTH,
    OP_LOG10,
    OP_LTH,
    OP_MAX0,
    OP_MIN0,
    ConstantModel,
    OperatorBase,
    ProductBase,
    SimpleBase,
)

EPS = 0.001


def similar(a, b, eps):
    return sum(abs(a - b)) < eps


def check_base(xtrain, model, fn):
    return similar(model.simulate(xtrain), fn(xtrain[:, 0]), EPS)


def test_simple_base(iris):
    xtrain = iris.iloc[:50, 0:2].values
    assert check_base(xtrain, SimpleBase(0, 1), lambda x: x)
    assert check_base(xtrain, SimpleBase(0, 2), lambda x: x ** 2)


def test_operator_base(iris):
    base_simple = SimpleBase(0, 1)
    base_abs = OperatorBase(base_simple, OP_ABS)
    base_max = OperatorBase(base_simple, OP_MAX0)
    base_min = OperatorBase(base_simple, OP_MIN0)
    base_log10 = OperatorBase(base_simple, OP_LOG10)
    base_gth = OperatorBase(base_simple, OP_GTH, 0.5)
    base_lth = OperatorBase(base_simple, OP_LTH, 0.5)

    xtrain = iris.iloc[:50, 0:2].values
    assert check_base(xtrain, base_abs, np.abs)
    assert check_base(xtrain, base_max, lambda x: np.clip(x, 0.0, INF))
    assert check_base(xtrain, base_min, lambda x: np.clip(x, -INF, 0.0))
    assert check_base(xtrain, base_log10, np.log10)
    assert check_base(xtrain, base_gth, lambda x: np.clip(0.5 - x, 0.0, INF))
    assert check_base(xtrain, base_lth, lambda x: np.clip(x - 0.5, 0.0, INF))


def test_product_base(iris):
    a = SimpleBase(0, 1)
    b = SimpleBase(0, 1)
    c = SimpleBase(0, 2)

    xtrain = iris.iloc[:50, 0:2].values
    assert check_base(xtrain, ProductBase(a, b), lambda x: x ** 2)
    assert check_base(xtrain, ProductBase(a, c), lambda x: x ** 3)


def test_constant_model(iris):
    xtrain = iris.iloc[:50, 0:2].values

    mu = xtrain[:, 0].mean()
    data = np.repeat(mu, xtrain.shape[0])
    assert similar(ConstantModel(mu, 0).simulate(xtrain), data, EPS)


def test_multi_ffx_model_factory(iris):
    np.random.seed(0)

    xtrain_pandas = iris.iloc[:50, 0:2]
    xtest_pandas = iris.iloc[51:100, 0:2]
    xtrain = xtrain_pandas.values
    ytrain = iris.iloc[:50, 2]
    xtest = xtest_pandas.values
    ytest = iris.iloc[51:100, 2]

    # Use numpy.ndarray
    models = ffx.run(xtrain, ytrain, xtest, ytest, iris.columns)
    assert abs(np.mean([model.test_nmse for model in models]) - 0.5821326214099275) < EPS

    # Use pandas.DataFrame
    models = ffx.run(xtrain_pandas, ytrain, xtest_pandas, ytest)
    assert abs(np.mean([model.test_nmse for model in models]) - 0.5821326214099275) < EPS
