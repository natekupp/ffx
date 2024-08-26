import abc

import numpy as np
import scipy
import six

from .constants import INF, OP_ABS, OP_GTH, OP_LOG10, OP_LTH, OP_MAX0, OP_MIN0
from .utils import coef_str


class Base(six.with_metaclass(abc.ABCMeta)):
    @abc.abstractmethod
    def simulate(self, X):
        pass

    @abc.abstractproperty
    def complexity(self):
        '''Return an integer measure of model complexity. It's intended to
        measure the number of nodes in the GP tree corresponding to
        the model. We assume the GP language includes: +, -, *, /,
        MAX0, MIN0, LOG10 but not GTH, LTH.  Thus, MAX0(x) returns the
        value max(0, x) but contributes only 1 + complexity(x) to the
        complexity count. GTH(thr, x) returns the value max(0, thr-x)
        but because it would be implemented in GP as MAX0(thr-x) it contributes
        3 + complexity(x) to the count.'''


class SimpleBase(Base):
    '''e.g. x4^2'''

    def __init__(self, var, exponent):
        self.var = var
        self.exponent = exponent

    def simulate(self, X):
        '''
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        '''
        return X[:, self.var] ** self.exponent

    def __str__(self):
        if self.exponent == 1:
            return 'x%d' % self.var
        else:
            return 'x%d^%g' % (self.var, self.exponent)

    @property
    def complexity(self):
        return 1 if self.exponent == 1 else 3


class OperatorBase(Base):
    '''e.g. log(x4^2)'''

    def __init__(self, simple_base, nonlin_op, thr=INF):
        '''
        @arguments
          simple_base -- SimpleBase
          nonlin_op -- one of OPS
          thr -- None or float -- depends on nonlin_op
        '''
        self.simple_base = simple_base
        self.nonlin_op = nonlin_op
        self.thr = thr

    def simulate(self, X):
        '''
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        '''
        op = self.nonlin_op
        ok = True
        y_lin = self.simple_base.simulate(X)

        if op == OP_ABS:
            ya = np.abs(y_lin)
        elif op == OP_MAX0:
            ya = np.clip(y_lin, 0.0, INF)
        elif op == OP_MIN0:
            ya = np.clip(y_lin, -INF, 0.0)
        elif op == OP_LOG10:
            # safeguard against: log() on values <= 0.0
            mn, mx = min(y_lin), max(y_lin)
            if mn <= 0.0 or np.isnan(mn) or mx == INF or np.isnan(mx):
                ok = False
            else:
                ya = np.log10(y_lin)
        elif op == OP_GTH:
            ya = np.clip(self.thr - y_lin, 0.0, INF)
        elif op == OP_LTH:
            ya = np.clip(y_lin - self.thr, 0.0, INF)
        else:
            raise 'Unknown op %d' % op

        if ok:  # could always do ** exp, but faster ways if exp is 0,1
            y = ya
        else:
            y = INF * np.ones(X.shape[0], dtype=float)
        return y

    def __str__(self):
        op = self.nonlin_op
        simple_s = str(self.simple_base)
        if op == OP_ABS:
            return 'abs(%s)' % simple_s
        elif op == OP_MAX0:
            return 'max(0, %s)' % simple_s
        elif op == OP_MIN0:
            return 'min(0, %s)' % simple_s
        elif op == OP_LOG10:
            return 'log10(%s)' % simple_s
        elif op == OP_GTH:
            return 'max(0,%s-%s)' % (coef_str(self.thr), simple_s)
        elif op == OP_LTH:
            return ('max(0,%s-%s)' % (simple_s, coef_str(self.thr))).replace('--', '+')
        else:
            raise 'Unknown op %d' % op

    @property
    def complexity(self):
        op = self.nonlin_op
        if op in [OP_ABS, OP_MAX0, OP_MIN0, OP_LOG10]:
            return 1 + self.simple_base.complexity
        elif op in [OP_GTH, OP_LTH]:
            return 3 + self.simple_base.complexity
        else:
            raise 'Unknown op %d' % op


class ProductBase(Base):
    '''e.g. x2^2 * log(x1^3)'''

    def __init__(self, base1, base2):
        self.base1 = base1
        self.base2 = base2

    def simulate(self, X):
        '''
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        '''
        yhat1 = self.base1.simulate(X)
        yhat2 = self.base2.simulate(X)
        return yhat1 * yhat2

    def __str__(self):
        return '%s * %s' % (self.base1, self.base2)

    @property
    def complexity(self):
        return 1 + self.base1.complexity + self.base2.complexity
