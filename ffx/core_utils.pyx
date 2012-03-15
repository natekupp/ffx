#cython: nonecheck=False
#cython: boundscheck=False
#cython: wraparound=False

# Moving things to Cython
import numpy, scipy
cimport numpy
cimport cython

#========================================================================================
#string functions

cpdef coefStr(double x):
    """Gracefully print a number to 3 significant digits.  See _testCoefStr in unit tests"""
    if x == 0.0:        s = '0'
    elif numpy.abs(x) < 1e-4: s = ('%.2e' % x).replace('e-0', 'e-')
    elif numpy.abs(x) < 1e-3: s = '%.6f' % x
    elif numpy.abs(x) < 1e-2: s = '%.5f' % x
    elif numpy.abs(x) < 1e-1: s = '%.4f' % x
    elif numpy.abs(x) < 1e0:  s = '%.3f' % x
    elif numpy.abs(x) < 1e1:  s = '%.2f' % x
    elif numpy.abs(x) < 1e2:  s = '%.1f' % x
    elif numpy.abs(x) < 1e4:  s = '%.0f' % x
    else:               s = ('%.2e' % x).replace('e+0', 'e')
    return s

cpdef basesStr(bases):
    """Pretty print list of bases"""
    return ', '.join([str(base) for base in bases])


#========================================================================================
#utility classes / functions
cdef extern from "math.h":
    bint isnan(double x)

cpdef yIsPoor(y):
    """Returns True if y is not usable"""
    return max(scipy.isinf(y)) or max(scipy.isnan(y))


# cpdef getParetoFront2d(numpy.ndarray[double, ndim=1]cost0s, 
#                        numpy.ndarray[int, ndim=1] cost1s):
#     """Extracts the 2D Pareto-optimal front from a 2D numpy array.
    
#     Parameters
#     ----------
#     data : numpy ndarray, or pandas.DataFrame
#         Data for which we want pareto-optimal front.
    
#     Examples
#     --------
#     p = getParetoFront(data)
    
#     """
#     cdef numpy.ndarray[int, ndim=1] mask = numpy.zeros(cost0s.shape[0], dtype=numpy.int32)

#     for i in range(cost0s.shape[0]):
#         for j in range(cost0s.shape[0]):
#             if i == j:
#                 continue
#             if cost0s[i] >= cost0s[j] and cost1s[i] >= cost1s[j]:
#                 mask[i] = 0
#     print mask
#     return numpy.nonzero(mask)[0]


@cython.wraparound(True)
cpdef nondominatedIndices2d(cost0s, cost1s):
    """
    @description
        Find indices of individuals that are on the nondominated 2-d tradeoff.

    @arguments
      cost0s -- 1d array of float [model_i] -- want to minimize this.  E.g. complexity.
      cost1s -- 1d array of float [model_i] -- want to minimize this too.  E.g. nmse.

    @return
      nondomI -- list of int -- nondominated indices; each is in range [0, #inds - 1]
                ALWAYS returns at least one entry if there is valid data        
    """ 
    #cost0s, cost1s = numpy.asarray(cost0s), numpy.asarray(cost1s)
    n_points = len(cost0s)
    assert n_points == len(cost1s)   

    if n_points == 0: #corner case
        return []

    #indices of cost0s sorted for ascending order  
    I = numpy.argsort(cost0s)

    #'cur' == best at this cost0s
    best_cost = [cost0s[I[0]], cost1s[I[0]]]
    best_cost_index = I[0]

    nondom_locs = []
    for i in xrange(n_points):
        loc = I[i] # traverse cost0s in ascending order
        if cost0s[loc] == best_cost[0]:
            if cost1s[loc] < best_cost[1]:
                best_cost_index = loc
                best_cost = [cost0s[loc], cost1s[loc]]
        else:   # cost0s[loc] > best_cost[0] because 
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




cpdef double nmse(numpy.ndarray[double, ndim=1] yhat, 
                  numpy.ndarray[double, ndim=1] y,
                  double min_y, 
                  double max_y):
    """
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

    Number to beat:  
        995    <<0.023>>    0.000    0.037    0.000 {core_utils_cython.nmse}
        995    <<0.020>>    0.000    0.037    0.000 {core_utils_cython.nmse}

    """
    cdef double y_range = max_y - min_y

    #base case: no entries
    if yhat.shape[0] == 0:
        return 0.0
    
    #base case: both yhat and y are constant, and same values
    if (max_y == min_y) and (numpy.max(yhat) == numpy.min(yhat) == numpy.max(y) == numpy.min(y)):
        return 0.0

    if (yhat.shape[0] != y.shape[0]) or (max_y < min_y):
        return numpy.Inf

    cdef double result = numpy.sqrt(1.0/yhat.shape[0]) * numpy.linalg.norm((yhat - y) / y_range)
    return numpy.Inf if isnan(result) else result



cpdef unbiasedXy(numpy.ndarray[double, ndim=2] Xin, 
                 numpy.ndarray[double, ndim=1] yin):

    """Make all input rows of X, and y, to have mean=0 stddev=1 """ 
    #unbiased X
    X_avgs, X_stds = Xin.mean(0), Xin.std(0)
    X_unbiased = (Xin - X_avgs) / X_stds
    
    #unbiased y
    y_avg, y_std = yin.mean(0), yin.std(0)
    y_unbiased = (yin - y_avg) / y_std
    
    return (X_unbiased, y_unbiased, X_avgs, X_stds, y_avg, y_std)
