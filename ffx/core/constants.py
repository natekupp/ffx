CONSIDER_INTER = True  # consider interactions?
CONSIDER_DENOM = True  # consider denominator?
CONSIDER_EXPON = True  # consider exponents?
CONSIDER_NONLIN = True  # consider abs() and log()?
CONSIDER_THRESH = True  # consider hinge functions?

INF = float('Inf')
# maximum time (s) for regularization update during pathwise learn.
MAX_TIME_REGULARIZE_UPDATE = 5

# GTH = Greater-Than Hinge function, LTH = Less-Than Hinge function
OP_ABS, OP_MAX0, OP_MIN0, OP_LOG10, OP_GTH, OP_LTH = 1, 2, 3, 4, 5, 6
