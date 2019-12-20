import pytest
from ffx.core.approach import Approach


def test_approach():
    a = Approach(use_inter=0, use_denom=0, use_expon=0, use_nonlin=0, use_thresh=0)
    assert str(a) == 'inter0 denom0 expon0 nonlin0 thresh0'
    assert a.num_feature_types == 0

    b = Approach(use_inter=1, use_denom=1, use_expon=1, use_nonlin=1, use_thresh=1)
    assert str(b) == 'inter1 denom1 expon1 nonlin1 thresh1'
    assert b.num_feature_types == 5

    with pytest.raises(AssertionError):
        Approach(use_inter='not an integer', use_denom=1, use_expon=1, use_nonlin=1, use_thresh=1)
