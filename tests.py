import numpy as np
import FFX, pandas

class TestFFX:
    def setUp(self): 
        self.EPS = 0.001
        self.data = pandas.read_csv('example-datasets/iris.csv')
        self.xtrain_pandas = self.data.ix[:50,0:2]
        self.xtest_pandas  = self.data.ix[51:100,0:2]
        self.xtrain        = self.xtrain_pandas.as_matrix()
        self.ytrain        = self.data.ix[:50,2]
        self.xtest         = self.xtest_pandas.as_matrix()
        self.ytest  = self.data.ix[51:100,2]

    def similar(self, a, b):
        return sum(abs(a - b)) < self.EPS

    # ----------------------------------------------------------------
    # Test bases
    # ----------------------------------------------------------------
    def runBase(self, model, data):
        return self.similar(model.simulate(self.xtrain), data)

    def testSimpleBase(self):
        assert self.runBase(FFX.SimpleBase(0,1), self.xtrain[:,0])
        assert self.runBase(FFX.SimpleBase(0,2), self.xtrain[:,0]**2)

    def testOperatorBase(self):
        a = FFX.SimpleBase(0,1)
        assert self.runBase(FFX.OperatorBase(a, FFX.OP_ABS),        np.abs(self.xtrain[:,0]))
        assert self.runBase(FFX.OperatorBase(a, FFX.OP_MAX0),       np.clip(self.xtrain[:,0], 0.0, FFX.INF))
        assert self.runBase(FFX.OperatorBase(a, FFX.OP_MIN0),       np.clip(self.xtrain[:,0], -FFX.INF, 0.0))
        assert self.runBase(FFX.OperatorBase(a, FFX.OP_LOG10),      np.log10(self.xtrain[:,0]))
        assert self.runBase(FFX.OperatorBase(a, FFX.OP_GTH, 0.5),   np.clip(0.5 - self.xtrain[:,0], 0.0, FFX.INF))
        assert self.runBase(FFX.OperatorBase(a, FFX.OP_LTH, 0.5),   np.clip(self.xtrain[:,0] - 0.5, 0.0, FFX.INF))

    def testProductBase(self):
        a = FFX.SimpleBase(0,1)
        b = FFX.SimpleBase(0,1)
        c = FFX.SimpleBase(0,2)
        assert self.runBase(FFX.ProductBase(a,b), self.xtrain[:,0]**2)
        assert self.runBase(FFX.ProductBase(a,c), self.xtrain[:,0]**3)


    # ----------------------------------------------------------------
    # Test models
    # ----------------------------------------------------------------
    def testConstantModel(self):
        mu = self.xtrain[:,0].mean()
        a  = FFX.ConstantModel(mu,0).simulate(self.xtrain)
        assert self.runBase(FFX.ConstantModel(mu,0),np.repeat(mu, self.xtrain.shape[0]))

    def testMultiFFXModelFactory(self):
        # Use numpy.ndarray
        models = FFX.MultiFFXModelFactory().build(self.xtrain, self.ytrain, self.xtest, self.ytest, self.data.columns)
        assert abs(np.mean([model.test_nmse for model in models]) - 0.4391323) < self.EPS

        # Use pandas.DataFrame
        models = FFX.MultiFFXModelFactory().build(self.xtrain_pandas, self.ytrain, self.xtest_pandas, self.ytest)
        assert abs(np.mean([model.test_nmse for model in models]) - 0.4391323) < self.EPS        




