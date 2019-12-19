import numpy as np


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
