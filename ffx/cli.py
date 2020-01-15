import os

import click
import numpy as np
import pandas as pd

from .api import run
from .time_utils import time_execution_scope


@click.group()
def main():
    '''Fast Function Extraction (FFX) toolkit.
    '''


@main.command()
@click.argument('x_file', type=click.Path(exists=True))
@click.argument('y_file', type=click.Path(exists=True))
def splitdata(x_file, y_file):
    '''Usage: ffx splitdata INPUTS_FILE[.csv/.txt] OUTPUTS_FILE[.csv/.txt]

    Given csv-formatted inputs and outputs files, splits them into training and testing data files
    of the form INPUTS_FILE_train.csv, OUTPUTS_FILE_train.csv, INPUTS_FILE_test.csv,
    OUTPUTS_FILE_test.csv.

    Sorts the data in ascending y.  Assigns every fourth value to test data; and rest to train data.

    In the csv files, there is one column for each sample point.  The inputs files have one row for
    each input variable.  The outputs files have just one row total, because the output is scalar.
    Values in a given row are separated by spaces.
    '''
    if not (x_file.endswith('.csv') or x_file.endswith('.txt')):
        print('INPUTS_FILE file \'%s\' needs to end with .csv or .txt.' % x_file)
        return

    if not (y_file.endswith('.csv') or y_file.endswith('.txt')):
        print('OUTPUTS_FILE file \'%s\' needs to end with .csv or .txt.' % y_file)
        return

    # create the target output filenames, and ensure they don't exist
    join = lambda n, prefix: os.path.join(os.path.dirname(n), prefix + os.path.basename(n))
    train_X_file = join(x_file, 'train_')
    train_y_file = join(y_file, 'train_')
    test_X_file = join(x_file, 'test_')
    test_y_file = join(y_file, 'test_')

    for newfile in [train_X_file, train_y_file, test_X_file, test_y_file]:
        if os.path.exists(newfile):
            print('New file \'%s\' exists, and should not. Early exit.' % newfile)
            return

    print('Begin ffx splitdata. INPUTS_FILE.csv=%s, OUTPUTS_FILE.csv=%s' % (x_file, y_file))

    X = pd.read_csv(x_file)  # [sample_i][var_i] : float
    y = pd.read_csv(y_file)  # [sample_i] : float

    if X.shape[0] != y.shape[0]:
        X = X.T
    assert X.shape[0] == y.shape[0], 'Error: X shape and y shape do not match. Early exit.'

    # create train/test data from X,y
    I = np.argsort(y)
    test_I, train_I = [], []
    for (loc, i) in enumerate(I):
        if loc % 4 == 0:
            test_I.append(i)
        else:
            train_I.append(i)

    train_X = np.take(X, train_I, 0)
    train_y = np.take(y, train_I)
    test_X = np.take(X, test_I, 0)
    test_y = np.take(y, test_I)

    print(
        'There will be %d samples in training data, and %d samples in test data'
        % (len(train_y), len(test_y))
    )

    delimiter = ',' if x_file.endswith('.csv') else '\t'
    np.savetxt(train_X_file, train_X, delimiter=delimiter)
    np.savetxt(train_y_file, train_y, delimiter=delimiter)
    np.savetxt(test_X_file, test_X, delimiter=delimiter)
    np.savetxt(test_y_file, test_y, delimiter=delimiter)

    print('Created these files:')
    print('  Training inputs:  %s' % train_X_file)
    print('  Training outputs: %s' % train_y_file)
    print('  Testing inputs:   %s' % test_X_file)
    print('  Testing outputs:  %s' % test_y_file)


@main.command()
@click.argument('samples-file', type=click.Path(exists=True))
def aboutdata(samples_file):
    '''Simply prints the number of variables and number of samples for the given file
    '''
    d = pd.read_csv(samples_file)
    print('Data file: %s' % samples_file)
    print('Number of input variables: %d' % d.shape[1])
    print('Number of input samples: %d' % d.shape[0])


@main.command()
@click.argument('train-x', type=click.Path(exists=True))
@click.argument('train-y', type=click.Path(exists=True))
@click.argument('test-x', type=click.Path(exists=True))
@click.argument('test-y', type=click.Path(exists=True))
@click.argument('varnames', type=click.Path())
def testffx(train_x, train_y, test_x, test_y, varnames):
    '''Usage: runffx test TRAIN_IN.csv TRAIN_OUT.csv TEST_IN.csv TEST_OUT.csv [VARNAMES.csv]

    - Builds a model from training data TRAIN_IN.csv and TRAIN_OUT.csv.
    - Computes & prints test nmse using test data TEST_IN.csv TEST_OUT.csv.
    - Also outputs the whole pareto optimal set of # bases vs. error in a .csv

    Arguments:
    TRAIN_IN.csv -- model input values for training data
    TRAIN_OUT.csv -- model output values for training data
    TEST_IN.csv -- model input values for testing data
    TEST_OUT.csv -- model output values for testing data
    VARNAMES.csv (optional) -- variable names.  One string for each variable name.

    In the training and test files, there is one column for each sample point.  The inputs
    files have one row for each input variable.  The outputs files have just one row total,
    because the output is scalar.  Values in a given row are separated by spaces.
    '''
    print('Begin ffx test.')

    # get X/y
    train_X, train_y, test_X, test_y = [pd.read_csv(f) for f in (train_x, train_y, test_x, test_y)]

    # get varnames
    varnames = pd.read_csv(varnames) if varnames else ['x%d' % i for i in range(train_X.shape[1])]

    # build models
    with time_execution_scope() as timer_result:
        models = run(train_X, train_y, test_X, test_y, varnames)

        output_csv = 'pareto_front_%s.csv' % str(int(timer_result.start_time))
        pd.DataFrame(
            [[model.numBases(), (model.test_nmse * 100.0), model] for model in models],
            columns=['Num Bases', 'Test error (%)', 'Model'],
        ).to_csv(output_csv, encoding='utf-8')

    print('Done.  Runtime: %.1f seconds.  Results are in: %s' % (timer_result.seconds, output_csv))


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
