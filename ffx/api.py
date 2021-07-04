'''api.py defines user interfaces to FFX. run() runs the complete method.
FFXRegressor is a Scikit-learn style regressor.
'''

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


def run(train_X, train_y, test_X, test_y, varnames=None, verbose=False):
    from .core import MultiFFXModelFactory

    return MultiFFXModelFactory().build(train_X, train_y, test_X, test_y, varnames, verbose)


class FFXRegressor(BaseEstimator, RegressorMixin):
    '''This class provides a Scikit-learn style estimator.'''

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
        # if X is a Pandas DataFrame, we don't have to pass in varnames.
        # otherwise we make up placeholders.
        if hasattr(X, 'columns'):
            varnames = None
        else:
            varnames = ["X%d" % i for i in range(len(X))]
        self.models_ = run(  # pylint: disable=attribute-defined-outside-init
            X, y, X, y, varnames=varnames
        )
        self.model_ = self.models_[-1]  # pylint: disable=attribute-defined-outside-init
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, accept_sparse=False)
        return self.model_.predict(X)

    def complexity(self):
        return self.model_.complexity()
