from .approach import Approach
from .constants import OP_ABS, OP_GTH, OP_LOG10, OP_LTH


class FFXBuildStrategy:
    '''All parameter settings.  Put magic numbers here.'''

    def __init__(self, approach):
        '''
        @arguments
          approach -- Approach object
        '''
        assert isinstance(approach, Approach)
        self.approach = approach

        self._num_alphas = 1000

        # final round will stop if either of these is hit
        self.final_target_train_nmse = 0.01  # 0.01 = 1%
        self.final_max_num_bases = 250

        # aggressive pruning (note: lasso has l1_ratio=1.0, ridge regression
        # has l1_ratio=0.0)
        self._l1_ratio = 0.95

        # eps -- Length of the path. eps=1e-3 means that alpha_min / alpha_max
        # = 1e-3.
        self._eps = 1e-70

        # will use all if 'nonlin1', else []
        self.all_nonlin_ops = [OP_ABS, OP_LOG10]

        # will use all if 'thresh1', else []
        self.all_threshold_ops = [OP_GTH, OP_LTH]
        self.num_thrs_per_var = 5

        # will use all if 'expon1', else [1.0]
        self.all_expr_exponents = [-1.0, -0.5, +0.5, +1.0]

    def include_interactions(self):
        return bool(self.approach.use_inter)

    def include_denominator(self):
        return bool(self.approach.use_denom)

    def expr_exponents(self):
        return self.all_expr_exponents if self.approach.use_expon else [1.0]

    def nonlin_ops(self):
        return self.all_nonlin_ops if self.approach.use_nonlin else []

    def threshold_ops(self):
        return self.all_threshold_ops if self.approach.use_thresh else []

    @property
    def eps(self):
        return self._eps

    @property
    def l1_ratio(self):
        return self._l1_ratio

    @property
    def num_alphas(self):
        return self._num_alphas
