import os

import pandas as pd
import pytest


@pytest.fixture(scope='session')
def iris():
    path = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(path, 'data/iris.csv'))
