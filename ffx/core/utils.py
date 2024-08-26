import math

import numpy as np
import scipy
from ffx.time_utils import timeout
from sklearn.linear_model import ElasticNet

from .constants import INF, MAX_TIME_REGULARIZE_UPDATE

# =========================================================================
# Revise linear_model.coordinate_descent.ElasticNet.fit() to handle when it hangs
# http://www.saltycrane.com/blog/2010/04/using-python-timeout-decorator-uploading-s3/


class ElasticNetWithTimeout(ElasticNet):

    # if this freezes, then exit with a TimeoutError
    @timeout(MAX_TIME_REGULARIZE_UPDATE)
    def fit(self, X, y, check_input=True):
        return ElasticNet.fit(self, X, y, check_input)


def nondominated_indices_2d(cost0s, cost1s):
    '''
    @description
        Find indices of individuals that are on the nondominated 2-d tradeoff.

    @arguments
      cost0s -- 1d array of float [model_i] -- want to minimize this.  E.g. complexity.
      cost1s -- 1d array of float [model_i] -- want to minimize this too.  E.g. nmse.

    @return
      nondomI -- list of int -- nondominated indices; each is in range [0, #inds - 1]
                ALWAYS returns at least one entry if there is valid data
    '''
    cost0s, cost1s = np.asarray(cost0s), np.asarray(cost1s)
    n_points = len(cost0s)
    assert n_points == len(cost1s)

    if n_points == 0:  # corner case
        return []

    # indices of cost0s sorted for ascending order
    I = np.argsort(cost0s)

    #'cur' == best at this cost0s
    best_cost = [cost0s[I[0]], cost1s[I[0]]]
    best_cost_index = I[0]

    nondom_locs = []
    for i in range(n_points):
        loc = I[i]  # traverse cost0s in ascending order
        if cost0s[loc] == best_cost[0]:
            if cost1s[loc] < best_cost[1]:
                best_cost_index = loc
                best_cost = [cost0s[loc], cost1s[loc]]
        else:  # cost0s[loc] > best_cost[0] because
            # loc indexes cost0s in ascending order
            if not nondom_locs:
                # initial value
                nondom_locs = [best_cost_index]
            elif best_cost[1] < cost1s[nondom_locs[-1]]:
                # if the current cost is lower than the last item
                # on the non-dominated list, add it's index to
                # the non-dominated list
                nondom_locs.append(best_cost_index)
            # set up "last tested value"
            best_cost_index = loc
            best_cost = [cost0s[loc], cost1s[loc]]

    if not nondom_locs:
        # if none are non-dominated, return the last one
        nondom_locs = [loc]
    elif best_cost[1] < cost1s[nondom_locs[-1]]:
        # if the current cost is lower than the last item
        # on the non-dominated list, add it's index to
        # the non-dominated list
        nondom_locs.append(best_cost_index)

    # return the non-dominated in sorted order
    nondomI = sorted(nondom_locs)
    return nondomI


def y_is_poor(y):
    '''Returns True if y is not usable'''
    return max(np.isinf(y)) or max(np.isnan(y))


def coef_str(x):
    '''Gracefully print a number to 3 significant digits.  See _testcoef_str in
    unit tests'''
    if x == 0.0:
        s = '0'
    elif np.abs(x) < 1e-4:
        s = ('%.2e' % x).replace('e-0', 'e-')
    elif np.abs(x) < 1e-3:
        s = '%.6f' % x
    elif np.abs(x) < 1e-2:
        s = '%.5f' % x
    elif np.abs(x) < 1e-1:
        s = '%.4f' % x
    elif np.abs(x) < 1e0:
        s = '%.3f' % x
    elif np.abs(x) < 1e1:
        s = '%.2f' % x
    elif np.abs(x) < 1e2:
        s = '%.1f' % x
    elif np.abs(x) < 1e4:
        s = '%.0f' % x
    else:
        s = ('%.2e' % x).replace('e+0', 'e')
    return s


def nmse(yhat, y, min_y, max_y):
    '''
    @description
        Calculates the normalized mean-squared error.

    @arguments
        yhat -- 1d array or list of floats -- estimated values of y
        y -- 1d array or list of floats -- true values
        min_y, max_y -- float, float -- roughly the min and max; they
          do not have to be the perfect values of min and max, because
          they're just here to scale the output into a roughly [0,1] range

    @return
        nmse -- float -- normalized mean-squared error
    '''
    # base case: no entries
    if len(yhat) == 0:
        return 0.0

    # base case: both yhat and y are constant, and same values
    if (max_y == min_y) and (max(yhat) == min(yhat) == max(y) == min(y)):
        return 0.0

    # main case
    assert max_y > min_y, 'max_y=%g was not > min_y=%g' % (max_y, min_y)
    yhat_a, y_a = np.asarray(yhat), np.asarray(y)
    y_range = float(max_y - min_y)

    result = math.sqrt(np.mean(((yhat_a - y_a) / y_range) ** 2))
    if np.isnan(result):
        return INF
    return result
