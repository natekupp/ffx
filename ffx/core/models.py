import numpy
from sklearn.base import RegressorMixin

from .constants import INF
from .utils import coef_str


class FFXModel(RegressorMixin):
    def __init__(self, coefs_n, bases_n, coefs_d, bases_d, varnames=None):
        """
        @arguments
          coefs_n -- 1d array of float -- coefficients for numerator.
          bases_n -- list of *Base -- bases for numerator
          coefs_d -- 1d array of float -- coefficients for denominator
          bases_d -- list of *Base -- bases for denominator
          varnames -- list of string
        """
        # preconditions
        # offset + numer_bases == numer_coefs
        assert 1 + len(bases_n) == len(coefs_n)
        assert len(bases_d) == len(coefs_d)  # denom_bases == denom_coefs

        # make sure that the coefs line up with their 'pretty' versions
        coefs_n = numpy.array([float(coef_str(coef)) for coef in coefs_n])
        coefs_d = numpy.array([float(coef_str(coef)) for coef in coefs_d])

        # reorder numerator bases from highest-to-lowest influence
        # -but keep offset 0th of course
        offset = coefs_n[0]
        coefs_n2 = coefs_n[1:]
        I = numpy.argsort(numpy.abs(coefs_n2))[::-1]
        coefs_n = [offset] + [coefs_n2[i] for i in I]
        bases_n = [bases_n[i] for i in I]

        # reorder denominator bases from highest-to-lowest influence
        I = numpy.argsort(numpy.abs(coefs_d))[::-1]
        coefs_d = [coefs_d[i] for i in I]
        bases_d = [bases_d[i] for i in I]

        # store values
        self.varnames = varnames
        self.coefs_n = coefs_n
        self.bases_n = bases_n
        self.coefs_d = coefs_d
        self.bases_d = bases_d

    def numBases(self):
        """Return total number of bases"""
        return len(self.bases_n) + len(self.bases_d)

    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        N = X.shape[0]

        # numerator
        y = numpy.zeros(N, dtype=float)
        y += self.coefs_n[0]
        for (coef, base) in zip(self.coefs_n[1:], self.bases_n):
            y += coef * base.simulate(X)

        # denominator
        if self.bases_d:
            denom_y = numpy.zeros(N, dtype=float)
            denom_y += 1.0
            for (coef, base) in zip(self.coefs_d, self.bases_d):
                denom_y += coef * base.simulate(X)
            y /= denom_y

        return y

    def predict(self, X):
        return self.simulate(X)

    def __str__(self):
        return self.str2()

    def str2(self, maxlen=100000):
        include_denom = bool(self.bases_d)

        s = ''
        # numerator
        if include_denom and len(self.coefs_n) > 1:
            s += '('
        numer_s = ['%s' % coef_str(self.coefs_n[0])]
        for (coef, base) in zip(self.coefs_n[1:], self.bases_n):
            numer_s += ['%s*%s' % (coef_str(coef), base)]
        s += ' + '.join(numer_s)
        if include_denom and len(self.coefs_n) > 1:
            s += ')'

        # denominator
        if self.bases_d:
            s += ' / ('
            denom_s = ['1.0']
            for (coef, base) in zip(self.coefs_d, self.bases_d):
                denom_s += ['%s*%s' % (coef_str(coef), base)]
            s += ' + '.join(denom_s)
            s += ')'

        # change xi to actual variable names
        for var_i in range(len(self.varnames) - 1, -1, -1):
            s = s.replace('x%d' % var_i, self.varnames[var_i])
        s = s.replace('+ -', '- ')

        # truncate long strings
        if len(s) > maxlen:
            s = s[:maxlen] + '...'

        return s

    def complexity(self):
        # Define complexity as the number of nodes needed in the
        # corresponding GP tree.

        # We have a leading constant, then for each base we have a
        # coefficient, a multiply, and a plus, plus the complexity of
        # the base itself.
        num_complexity = 1 + sum(3 + b.complexity for b in self.bases_n)
        if self.bases_d:
            denom_complexity = 1 + sum(3 + b.complexity for b in self.bases_d)
            # add 1 for the division
            return num_complexity + 1 + denom_complexity
        else:
            return num_complexity


class ConstantModel(RegressorMixin):
    """e.g. 3.2"""

    def __init__(self, constant, numvars):
        """
        @description
            Constructor.

        @arguments
            constant -- float -- constant value returned by this model
            numvars -- int -- number of input variables to this model
        """
        self.constant = float(constant)
        self.numvars = numvars

    def numBases(self):
        """Return total number of bases"""
        return 0

    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        N = X.shape[0]
        if numpy.isnan(self.constant):  # corner case
            yhat = numpy.array([INF] * N)
        else:  # typical case
            yhat = numpy.ones(N, dtype=float) * self.constant
        return yhat

    def predict(self, X):
        return self.simulate(X)

    def __str__(self):
        return self.str2()

    def str2(self, dummy_arg=None):  # pylint: disable=unused-argument
        return coef_str(self.constant)

    def complexity(self):
        return 1
