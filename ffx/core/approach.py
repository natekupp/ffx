from collections import namedtuple


class Approach(namedtuple('_Approach', 'use_inter use_denom use_expon use_nonlin use_thresh')):
    def __new__(cls, use_inter, use_denom, use_expon, use_nonlin, use_thresh):
        assert set([use_inter, use_denom, use_expon, use_nonlin, use_thresh]).issubset([0, 1])
        return super(Approach, cls).__new__(
            cls, use_inter, use_denom, use_expon, use_nonlin, use_thresh
        )

    @property
    def num_feature_types(self):
        '''How many types of features does this approach consider?
        '''
        return sum(self._asdict().values())

    def __repr__(self):
        return 'inter%d denom%d expon%d nonlin%d thresh%d' % (
            self.use_inter,
            self.use_denom,
            self.use_expon,
            self.use_nonlin,
            self.use_thresh,
        )
