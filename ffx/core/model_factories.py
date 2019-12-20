import math
import sys
import time

import numpy
import pandas as pd
from ffx.time_utils import TimeoutError  # pylint: disable=redefined-builtin

from .approach import Approach
from .bases import OperatorBase, ProductBase, SimpleBase
from .build_strategy import FFXBuildStrategy
from .constants import (
    CONSIDER_DENOM,
    CONSIDER_EXPON,
    CONSIDER_INTER,
    CONSIDER_NONLIN,
    CONSIDER_THRESH,
    OP_ABS,
    OP_MAX0,
    OP_MIN0,
)
from .models import ConstantModel, FFXModel
from .utils import ElasticNetWithTimeout, nmse, nondominated_indices_2d, y_is_poor


class MultiFFXModelFactory:
    def build(self, train_X, train_y, test_X, test_y, varnames=None, verbose=False):
        """
        @description
          Builds FFX models at many different settings, then merges the results
          into a single Pareto Optimal Set.

        @argument
          train_X -- 2d array of [sample_i][var_i] : float -- training inputs
          train_y -- 1d array of [sample_i] : float -- training outputs
          test_X -- 2d array -- testing inputs
          test_y -- 1d array -- testing outputs
          varnames -- list of string -- variable names

        @return
          models -- list of FFXModel -- Pareto-optimal set of models
        """

        if isinstance(train_X, pd.DataFrame):
            varnames = train_X.columns
            train_X = train_X.to_numpy()
            test_X = test_X.to_numpy()
        if isinstance(train_X, numpy.ndarray) and varnames is None:
            raise Exception('varnames required for numpy.ndarray')

        if verbose:
            print(
                'Build(): begin. {2} variables, {1} training samples, {0} test samples'.format(
                    test_X.shape[0], *train_X.shape
                )
            )

        models = []
        min_y = min(min(train_y), min(test_y))
        max_y = max(max(train_y), max(test_y))

        # build all combinations of approaches, except for (a) features we don't consider
        # and (b) too many features at once
        approaches = []
        if verbose:
            print("Learning Approaches Considered:")
            print("=========================================")
            print("Inter   Denom   Expon   Nonlin  Threshold")
            print("=========================================")
        if CONSIDER_INTER:
            inters = [1]  # inter=0 is a subset of inter=1
        else:
            inters = [0]
        for inter in inters:
            for denom in [0, 1]:
                if denom == 1 and not CONSIDER_DENOM:
                    continue
                for expon in [0, 1]:
                    if expon == 1 and not CONSIDER_EXPON:
                        continue
                    if expon == 1 and inter == 1:
                        continue  # never need both exponent and inter
                    for nonlin in [0, 1]:
                        if nonlin == 1 and not CONSIDER_NONLIN:
                            continue
                        for thresh in [0, 1]:
                            if thresh == 1 and not CONSIDER_THRESH:
                                continue
                            approach = Approach(inter, denom, expon, nonlin, thresh)
                            if approach.num_feature_types >= 4:
                                continue  # not too many features at once
                            approaches.append(approach)
                            if verbose:
                                print(approach)

        for (i, approach) in enumerate(approaches):
            if verbose:
                print('-' * 200)
                print(
                    'Build with approach %d/%d (%s): begin'
                    % (i + 1, len(approaches), str(approach))
                )
            ss = FFXBuildStrategy(approach)

            next_models = FFXModelFactory().build(train_X, train_y, ss, varnames, verbose)

            # set test_nmse on each model
            for model in next_models:
                test_yhat = model.simulate(test_X)
                model.test_nmse = nmse(  # pylint: disable=attribute-defined-outside-init
                    test_yhat, test_y, min_y, max_y
                )

            # pareto filter
            if verbose:
                print('  STEP 3: Nondominated filter')
            next_models = self._nondominatedModels(next_models)
            models += next_models
            if verbose:
                print(
                    'Build with approach %d/%d (%s): done.  %d model(s).'
                    % (i + 1, len(approaches), str(approach), len(next_models))
                )
                print('Models:')
                for model in next_models:
                    print(
                        "num_bases=%d, test_nmse=%.6f, model=%s"
                        % (model.numBases(), model.test_nmse, model.str2(500))
                    )

        # final pareto filter
        models = self._nondominatedModels(models)

        # log nondominated models
        if verbose:
            print('=' * 200)
            print('%d nondominated models (wrt test error & num. bases):' % len(models))
            for (i, model) in enumerate(models):
                print(
                    "Nondom model %d/%d: num_bases=%d, test_nmse=%.6f, model=%s"
                    % (i + 1, len(models), model.numBases(), model.test_nmse, model.str2(500))
                )

        return models

    def _nondominatedModels(self, models):
        test_nmses = [model.test_nmse for model in models]
        num_bases = [model.numBases() for model in models]
        I = nondominated_indices_2d(test_nmses, num_bases)
        models = [models[i] for i in I]

        I = numpy.argsort([model.numBases() for model in models])
        models = [models[i] for i in I]

        return models


class FFXModelFactory:
    def build(self, X, y, ss, varnames=None, verbose=False):
        """
        @description
          Build FFX models at the settings of input solution strategy 'ss'.

        @argument
          X -- 2d array of [var_i][sample_i] : float -- training inputs
          y -- 1d array of [sample_i] : float -- training outputs
          varnames -- list of string -- variable names
          ss -- FFXSolutionStrategy

        @return
          models -- list of FFXModel -- Pareto-optimal set of models
        """
        if isinstance(X, pd.DataFrame):
            varnames = X.columns
            X = X.to_numpy()
        if isinstance(X, numpy.ndarray) and varnames is None:
            raise Exception('varnames required for numpy.ndarray')

        if X.ndim == 1:
            self.nrow, self.ncol = len(X), 1  # pylint: disable=attribute-defined-outside-init
        elif X.ndim == 2:
            self.nrow, self.ncol = X.shape  # pylint: disable=attribute-defined-outside-init
        else:
            raise Exception('X is wrong dimensionality.')

        y = numpy.asarray(y)
        if self.nrow != len(y):
            raise Exception('X sample count and y sample count do not match')

        # if y has shape (N, 1) then we reshape to just (N,)
        if len(y.shape) > 1:
            assert y.shape[1] == 1
            y = numpy.reshape(y, (y.shape[0],))

        if self.ncol == 0:
            print('  Corner case: no input vars, so return a ConstantModel')
            return [ConstantModel(y.mean(), 0)]

        # Main work...

        # build up each combination of all {var_i} x {op_j}, except for
        # when a combination is unsuccessful
        if verbose:
            print('  STEP 1A: Build order-1 bases: begin')
        order1_bases = []
        for var_i in range(self.ncol):
            for exponent in ss.expr_exponents():
                if exponent == 0.0:
                    continue

                #'lin' version of base
                simple_base = SimpleBase(var_i, exponent)
                lin_yhat = simple_base.simulate(X)
                # checking exponent is a speedup
                if exponent in [1.0, 2.0] or not y_is_poor(lin_yhat):
                    order1_bases.append(simple_base)

                    # add e.g. OP_ABS, OP_MAX0, OP_MIN0, OP_LOG10
                    for nonlin_op in ss.nonlin_ops():
                        # ignore cases where op has no effect
                        if nonlin_op == OP_ABS and exponent in [-2, +2]:
                            continue
                        if nonlin_op == OP_MAX0 and min(lin_yhat) >= 0:
                            continue
                        if nonlin_op == OP_MIN0 and max(lin_yhat) <= 0:
                            continue

                        nonsimple_base = OperatorBase(simple_base, nonlin_op, None)

                        # easy access when considering interactions
                        nonsimple_base.var = var_i  # pylint: disable=attribute-defined-outside-init

                        nonlin_yhat = nonsimple_base.simulate(X)
                        if numpy.all(nonlin_yhat == lin_yhat):
                            continue  # op has no effect
                        if not y_is_poor(nonlin_yhat):
                            order1_bases.append(nonsimple_base)

                    # add e.g. OP_GTH, OP_LTH
                    if exponent == 1.0 and ss.threshold_ops():
                        minx, maxx = min(X[:, var_i]), max(X[:, var_i])
                        rangex = maxx - minx
                        if rangex > 0:
                            stepx = 0.8 * rangex / float(ss.num_thrs_per_var + 1)
                            thrs = numpy.arange(
                                minx + 0.2 * rangex, maxx - 0.2 * rangex + 0.1 * rangex, stepx
                            )
                        else:
                            continue
                        for threshold_op in ss.threshold_ops():
                            for thr in thrs:
                                nonsimple_base = OperatorBase(simple_base, threshold_op, thr)
                                # easy access when considering interactions
                                nonsimple_base.var = (  # pylint: disable=attribute-defined-outside-init
                                    var_i
                                )
                                order1_bases.append(nonsimple_base)
        if verbose:
            print(
                '  STEP 1A: Build order-1 bases: done.  Have %d order-1 bases.' % len(order1_bases)
            )

        var1_models = None
        if ss.include_interactions():
            # find base-1 influences
            if verbose:
                print('  STEP 1B: Find order-1 base infls: begin')
            max_num_bases = len(order1_bases)  # no limit
            target_train_nmse = 0.01
            models = self._basesToModels(
                ss, varnames, order1_bases, X, y, max_num_bases, target_train_nmse, verbose
            )
            if models is None:  # fit failed.
                model = ConstantModel(y[0], 0)
                return [model]
            var1_models = models

            # use most-explaining model (which also has the max num bases)
            model = models[-1]
            order1_bases = model.bases_n + model.bases_d

            if len(order1_bases) == 0:  # the most-explaining model is a constant model
                model = ConstantModel(y[0], 0)
                return [model]

            # order bases by influence
            order1_infls = numpy.abs(list(model.coefs_n[1:]) + list(model.coefs_d))  # influences
            I = numpy.argsort(-1 * order1_infls)
            order1_bases = [order1_bases[i] for i in I]
            if verbose:
                print('  STEP 1B: Find order-1 base infls: done')

            # don't let inter coeffs swamp linear ones; but handle more when n
            # small
            n_order1_bases = len(order1_bases)
            max_n_order2_bases = 3 * math.sqrt(n_order1_bases)  # default
            max_n_order2_bases = max(max_n_order2_bases, 10)  # lower limit
            max_n_order2_bases = max(max_n_order2_bases, 2 * n_order1_bases)  # ""
            if ss.include_denominator():
                max_n_order2_bases = min(max_n_order2_bases, 4000)  # upper limit
            else:
                max_n_order2_bases = min(max_n_order2_bases, 8000)  # ""

            # build up order2 bases
            if verbose:
                print('  STEP 1C: Build order-2 bases: begin')

            #  -always have all xi*xi terms
            order2_bases = []
            order2_basetups = set()  # set of (basei_id, basej_id) tuples
            for i, basei in enumerate(order1_bases):
                if basei.__class__ != SimpleBase:
                    continue  # just xi
                if basei.exponent != 1.0:
                    continue  # just exponent==1

                combined_base = SimpleBase(basei.var, 2)
                order2_bases.append(combined_base)

                tup = (id(basei), id(basei))
                order2_basetups.add(tup)

            # -then add other terms
            for max_base_i in range(len(order1_bases)):
                for i in range(max_base_i):
                    basei = order1_bases[i]
                    for j in range(max_base_i):
                        if j >= i:
                            continue  # disallow mirror image
                        basej = order1_bases[j]
                        tup = (id(basei), id(basej))
                        if tup in order2_basetups:
                            continue  # no duplicate pairs
                        combined_base = ProductBase(basei, basej)
                        order2_bases.append(combined_base)
                        order2_basetups.add(tup)

                        if len(order2_bases) >= max_n_order2_bases:
                            break  # for j
                    if len(order2_bases) >= max_n_order2_bases:
                        break  # for i
                if len(order2_bases) >= max_n_order2_bases:
                    break  # for max_base_i

            if verbose:
                print(
                    '  STEP 1C: Build order-2 bases: done.  Have %d order-2'
                    ' bases.' % len(order2_bases)
                )
            bases = order1_bases + order2_bases
        else:
            bases = order1_bases

        # all bases. Stop based on target nmse, not number of bases
        if verbose:
            print('  STEP 2: Regress on all %d bases: begin.' % len(bases))
        var2_models = self._basesToModels(
            ss, varnames, bases, X, y, ss.final_max_num_bases, ss.final_target_train_nmse, verbose
        )
        if verbose:
            print('  STEP 2: Regress on all %d bases: done.' % len(bases))

        # combine models having 1-var with models having 2-vars
        if var1_models is None and var2_models is None:
            models = []
        elif var1_models is None and var2_models is not None:
            models = var2_models
        elif var1_models is not None and var2_models is None:
            models = var1_models
        else:  # var1_models is not None and var2_models is not None
            models = var1_models + var2_models

        # add constant; done
        models = [ConstantModel(numpy.mean(y), X.shape[0])] + models
        return models

    def _basesToModels(self, ss, varnames, bases, X, y, max_num_bases, target_train_nmse, verbose):
        # compute regress_X
        if ss.include_denominator():
            regress_X = numpy.zeros((self.nrow, len(bases) * 2), dtype=float)
        else:
            regress_X = numpy.zeros((self.nrow, len(bases)), dtype=float)
        for i, base in enumerate(bases):
            base_y = base.simulate(X)
            regress_X[:, i] = base_y  # numerators
            if ss.include_denominator():
                regress_X[:, len(bases) + i] = -1.0 * base_y * y  # denominators

        # compute models.
        models = self._pathwiseLearn(
            ss, varnames, bases, X, regress_X, y, max_num_bases, target_train_nmse, verbose
        )
        return models

    def _pathwiseLearn(
        self,
        ss,
        varnames,
        bases,
        X_orig,  # pylint: disable=unused-argument
        X_orig_regress,
        y_orig,
        max_num_bases,
        target_nmse,
        verbose=False,
        **fit_params
    ):
        """Adapted from enet_path() in sklearn.linear_model.
        http://scikit-learn.sourceforge.net/modules/linear_model.html
        Compute Elastic-Net path with coordinate descent.
        Returns list of model (or None if failure)."""
        if verbose:
            print('    Pathwise learn: begin. max_num_bases=%d' % max_num_bases)
        max_iter = 5000  # default 5000. magic number.

        # Condition X and y:
        # -"unbias" = rescale so that (mean=0, stddev=1) -- subtract each row's
        #             mean, then divide by stddev
        # -X transposed
        # -X as fortran array
        (X_unbiased, y_unbiased, X_avgs, X_stds, y_avg, y_std) = self._unbiasedXy(
            X_orig_regress, y_orig
        )
        # make data contiguous in memory
        X_unbiased = numpy.asfortranarray(X_unbiased)

        n_samples = X_unbiased.shape[0]
        vals = numpy.dot(X_unbiased.T, y_unbiased)
        vals = [val for val in vals if not numpy.isnan(val)]
        if vals:
            alpha_max = numpy.abs(max(vals) / (n_samples * ss.l1_ratio))
        else:
            alpha_max = 1.0  # backup: pick a value from the air

        # alphas = lotsa alphas at beginning, and usual rate for rest
        st, fin = numpy.log10(alpha_max * ss.eps), numpy.log10(alpha_max)
        alphas1 = numpy.logspace(st, fin, num=ss.num_alphas * 10)[::-1][: ss.num_alphas // 4]
        alphas2 = numpy.logspace(st, fin, num=ss.num_alphas)
        alphas = sorted(set(alphas1).union(alphas2), reverse=True)

        if 'precompute' not in fit_params or fit_params['precompute'] is True:
            fit_params['precompute'] = numpy.dot(X_unbiased.T, X_unbiased)
            # if not 'Xy' in fit_params or fit_params['Xy'] is None:
            #    fit_params['Xy'] = numpy.dot(X_unbiased.T, y_unbiased)

        models = []  # build this up
        nmses = []  # for detecting stagnation
        cur_unbiased_coefs = None  # init values for coefs
        start_time = time.time()
        for (alpha_i, alpha) in enumerate(alphas):
            # compute (unbiased) coefficients. Recall that mean=0 so no
            # intercept needed
            clf = ElasticNetWithTimeout(
                alpha=alpha,
                l1_ratio=ss.l1_ratio,
                fit_intercept=False,
                max_iter=max_iter,
                **fit_params
            )
            try:
                clf.fit(X_unbiased, y_unbiased)
            except TimeoutError:
                print('    Regularized update failed. Returning None')
                return None  # failure
            except ValueError:
                print('    Regularized update failed with ValueError.')
                print('    X_unbiased:')
                print(X_unbiased)
                print('    y_unbiased:')
                print(y_unbiased)
                sys.exit(1)

            cur_unbiased_coefs = clf.coef_.copy()
            if cur_unbiased_coefs.shape == tuple():
                # This happens when we have only one variable because
                # ElasticNet calls numpy.squeeze(), which reduces a
                # single element array to a 0-d array. That would
                # crash us below in list(cur_unbiased_coefs). We just
                # undo the squeeze.
                cur_unbiased_coefs = cur_unbiased_coefs.reshape((1,))

            # compute model; update models
            #  -"rebias" means convert from (mean=0, stddev=1) to original (mean, stddev)
            coefs = self._rebiasCoefs(
                [0.0] + list(cur_unbiased_coefs), X_stds, X_avgs, y_std, y_avg
            )
            (coefs_n, bases_n, coefs_d, bases_d) = self._allocateToNumerDenom(ss, bases, coefs)
            model = FFXModel(coefs_n, bases_n, coefs_d, bases_d, varnames)
            models.append(model)

            # update nmses
            nmse_unbiased = nmse(
                numpy.dot(cur_unbiased_coefs, X_unbiased.T),
                y_unbiased,
                min(y_unbiased),
                max(y_unbiased),
            )
            nmses.append(nmse_unbiased)

            # log
            num_bases = len(numpy.nonzero(cur_unbiased_coefs)[0])
            if verbose and ((alpha_i == 0) or (alpha_i + 1) % 50 == 0):
                print(
                    '      alpha %d/%d (%3e): num_bases=%d, nmse=%.6f, time %.2f s'
                    % (
                        alpha_i + 1,
                        len(alphas),
                        alpha,
                        num_bases,
                        nmse_unbiased,
                        time.time() - start_time,
                    )
                )

            # maybe stop
            if numpy.isinf(nmses[-1]):
                if verbose:
                    print('    Pathwise learn: Early stop because nmse is inf')
                return None
            if nmse_unbiased < target_nmse:
                if verbose:
                    print('    Pathwise learn: Early stop because nmse < target')
                return models
            if num_bases > max_num_bases:
                if verbose:
                    print('    Pathwise learn: Early stop because num bases > %d' % max_num_bases)
                return models
            if len(nmses) > 15 and round(nmses[-1], 4) == round(nmses[-15], 4):
                if verbose:
                    print('    Pathwise learn: Early stop because nmses stagnated')
                return models
        if verbose:
            print('    Pathwise learn: done')
        return models

    def _allocateToNumerDenom(self, ss, bases, coefs):
        """Prune out nonzero coefficients/bases.  Allocate to numerator vs. denominator."""
        if ss.include_denominator():
            # offset + numer_bases + denom_bases
            assert 1 + len(bases) + len(bases) == len(coefs)
            n_bases = len(bases)
            coefs_n = [coefs[0]] + [coef for coef in coefs[1 : n_bases + 1] if coef != 0]
            bases_n = [base for (base, coef) in zip(bases, coefs[1 : n_bases + 1]) if coef != 0]
            coefs_d = [coef for coef in coefs[n_bases + 1 :] if coef != 0]
            bases_d = [base for (base, coef) in zip(bases, coefs[n_bases + 1 :]) if coef != 0]

        else:
            # offset + numer_bases + denom_bases
            assert 1 + len(bases) == len(coefs)
            coefs_n = [coefs[0]] + [coef for coef in coefs[1:] if coef != 0]
            bases_n = [base for (base, coef) in zip(bases, coefs[1:]) if coef != 0]
            coefs_d = []
            bases_d = []

        return (coefs_n, bases_n, coefs_d, bases_d)

    def _unbiasedXy(self, Xin, yin):
        """Make all input rows of X, and y, to have mean=0 stddev=1 """
        # unbiased X
        X_avgs, X_stds = Xin.mean(0), Xin.std(0)
        # if any stddev was 0, overwrite with 1 to prevent divide by
        # zero. Because we then return the overwritten value,
        # _rebiasCoefs will end up with the right rebiased values. Same
        # for y below.
        numpy.copyto(X_stds, 1.0, where=~(X_stds > 0.0))
        X_unbiased = (Xin - X_avgs) / X_stds

        # unbiased y
        y_avg, y_std = yin.mean(0), yin.std(0)
        # if stddev was 0, overwrite with 1
        if not y_std > 0.0:
            y_std = 1.0
        y_unbiased = (yin - y_avg) / y_std

        assert numpy.all(numpy.isfinite(X_unbiased))
        assert numpy.all(numpy.isfinite(y_unbiased))

        return (X_unbiased, y_unbiased, X_avgs, X_stds, y_avg, y_std)

    def _rebiasCoefs(self, unbiased_coefs, X_stds, X_avgs, y_std, y_avg):
        """Given the coefficients that were learned using unbiased training data, rebias the
        coefficients so that they are at the scale of the real training X and y."""
        # preconditions
        assert unbiased_coefs is not None
        assert len(unbiased_coefs) == (len(X_stds) + 1) == (len(X_avgs) + 1), (
            len(unbiased_coefs),
            (len(X_stds) + 1),
            (len(X_avgs) + 1),
        )

        # main work
        n = len(X_stds)
        coefs = numpy.zeros(n + 1, dtype=float)
        coefs[0] = unbiased_coefs[0] * y_std + y_avg
        for j in range(1, n + 1):
            coefs[j] = unbiased_coefs[j] * y_std / X_stds[j - 1]
            coefs[0] -= coefs[j] * X_avgs[j - 1]
        return coefs
