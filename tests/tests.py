import numpy as np
import ffx, pandas
from ffx.core import *

def similar(a, b, eps):
    return sum(abs(a - b)) < eps

class TestFFX:
    def setUp(self): 
        self.EPS = 0.001
        self.data = pandas.read_csv('data/iris.csv')
        self.xtrain_pandas = self.data.ix[:50,0:2]
        self.xtest_pandas  = self.data.ix[51:100,0:2]
        self.xtrain        = self.xtrain_pandas.as_matrix()
        self.ytrain        = self.data.ix[:50,2]
        self.xtest         = self.xtest_pandas.as_matrix()
        self.ytest  = self.data.ix[51:100,2]

    # ----------------------------------------------------------------
    # Test bases
    # ----------------------------------------------------------------
    def checkBase(self, model, fn):
        return similar(model.simulate(self.xtrain), fn(self.xtrain[:,0]), self.EPS)

    def testSimpleBase(self):
        assert self.checkBase(SimpleBase(0,1), lambda x: x)
        assert self.checkBase(SimpleBase(0,2), lambda x: x**2)

    def testOperatorBase(self):    
        base_simple = SimpleBase(0,1)
        base_abs    = OperatorBase(base_simple, OP_ABS)
        base_max    = OperatorBase(base_simple, OP_MAX0)
        base_min    = OperatorBase(base_simple, OP_MIN0)
        base_log10  = OperatorBase(base_simple, OP_LOG10)
        base_gth    = OperatorBase(base_simple, OP_GTH, 0.5)
        base_lth    = OperatorBase(base_simple, OP_LTH, 0.5)
        
        assert self.checkBase(base_abs,   np.abs)
        assert self.checkBase(base_max,   lambda x: np.clip(x, 0.0, INF))
        assert self.checkBase(base_min,   lambda x: np.clip(x, -INF, 0.0))
        assert self.checkBase(base_log10, np.log10)
        assert self.checkBase(base_gth,   lambda x: np.clip(0.5 - x, 0.0, INF))
        assert self.checkBase(base_lth,   lambda x: np.clip(x - 0.5, 0.0, INF))

    def testProductBase(self):
        a = SimpleBase(0,1)
        b = SimpleBase(0,1)
        c = SimpleBase(0,2)
        assert self.checkBase(ProductBase(a,b), lambda x: x**2)
        assert self.checkBase(ProductBase(a,c), lambda x: x**3)

    # ----------------------------------------------------------------
    # Test constant model
    # ----------------------------------------------------------------
    def testConstantModel(self):
        mu   = self.xtrain[:,0].mean()
        data = np.repeat(mu, self.xtrain.shape[0])
        assert similar(ConstantModel(mu,0).simulate(self.xtrain), data, self.EPS)

    # ----------------------------------------------------------------
    # Test FFX API
    # ----------------------------------------------------------------
    def testMultiFFXModelFactory(self):
        # Use numpy.ndarray
        models = ffx.run(self.xtrain, self.ytrain, self.xtest, self.ytest, self.data.columns)
        assert abs(np.mean([model.test_nmse for model in models]) - 0.4391323) < self.EPS

        # Use pandas.DataFrame
        models = ffx.run(self.xtrain_pandas, self.ytrain, self.xtest_pandas, self.ytest)
        assert abs(np.mean([model.test_nmse for model in models]) - 0.4391323) < self.EPS        





