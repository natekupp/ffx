"""FFX.py v1.3 (Sept 16, 2011)
This module implements the Fast Function Extraction (FFX) algorithm.

Reference: Trent McConaghy, FFX: Fast, Scalable, Deterministic Symbolic Regression Technology, Genetic Programming Theory and Practice IX, Edited by R. Riolo, E. Vladislavleva, and J. Moore, Springer, 2011.  http://www.trent.st/ffx


HOW TO USE THIS MODULE:

Easiest to use by calling runffx.py.  Its code has example usage patterns.

The main routines are:
  models = MultiFFXModelFactory().build(train_X, train_y, test_X, test_y, varnames)
  yhat = model.simulate(X)
  print model

Can expand / restrict the set of functions via the user-changeable constants (right below licence).
"""

"""
FFX Software Licence Agreement (like BSD, but adapted for non-commercial gain only)

Copyright (c) 2011, Solido Design Automation Inc.  Authored by Trent McConaghy.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    * Usage does not involve commercial gain. 
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of the associated institutions nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

For permissions beyond the scope of this license, please contact Trent McConaghy (trentmc@solidodesign.com).

THIS SOFTWARE IS PROVIDED BY THE DEVELOPERS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE DEVELOPERS OR THEIR INSTITUTIONS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

Patent pending.
"""

#user-changeable constants
CONSIDER_INTER = True #consider interactions?
CONSIDER_DENOM = True #consider denominator?
CONSIDER_EXPON = True #consider exponents?
CONSIDER_NONLIN = True #consider abs() and log()?
CONSIDER_THRESH = True #consider hinge functions?

#======================================================================================
import copy, itertools, math, signal, time, types, pandas

#3rd party dependencies
import numpy
import scipy
from scikits.learn.linear_model.coordinate_descent import ElasticNet

INF = float('Inf')
MAX_TIME_REGULARIZE_UPDATE = 5 #maximum time (s) for regularization update during pathwise learn.

#GTH = Greater-Than Hinge function, LTH = Less-Than Hinge function
OP_ABS, OP_MAX0, OP_MIN0, OP_LOG10, OP_GTH, OP_LTH = 1, 2, 3, 4, 5, 6

def _approachStr(approach):
    assert len(approach) == 5
    assert set(approach).issubset([0,1])
    return 'inter%d denom%d expon%d nonlin%d thresh%d' % \
        (approach[0], approach[1], approach[2], approach[3], approach[4])

#========================================================================================
#strategy 
class FFXBuildStrategy(object):
    """All parameter settings.  Put magic numbers here."""
    
    def __init__(self, approach):
        """
        @arguments
          approach -- 5-d list of [use_inter, use_denom, use_expon, use_nonlin, use_thresh]
        """
        assert len(approach) == 5
        assert set(approach).issubset([0,1])
        self.approach = approach 
        
        self.num_alphas = 1000

        #final round will stop if either of these is hit
        self.final_target_train_nmse = 0.01 #0.01 = 1%
        self.final_max_num_bases = 250 #

        self._rho = 0.95 #aggressive pruning (note: lasso has rho=1.0, ridge regression has rho=0.0)
        
        #  eps -- Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3.
        self._eps = 1e-70

        #will use all if 'nonlin1', else []
        self.all_nonlin_ops = [OP_ABS, OP_LOG10] 

        #will use all if 'thresh1', else []
        self.all_threshold_ops = [OP_GTH, OP_LTH] 
        self.num_thrs_per_var = 5

        #will use all if 'expon1', else [1.0]
        self.all_expr_exponents = [-1.0, -0.5, +0.5, +1.0]

    def includeInteractions(self):
        return bool(self.approach[0])

    def includeDenominator(self):
        return bool(self.approach[1])

    def exprExponents(self):
        if self.approach[2]: return self.all_expr_exponents
        else:                return [1.0]

    def nonlinOps(self):
        if self.approach[3]: return self.all_nonlin_ops
        else:                return []

    def thresholdOps(self):
        if self.approach[4]: return self.all_threshold_ops
        else:                return []

    def eps(self):
        return self._eps

    def rho(self):
        return self._rho

    def numAlphas(self):
        return self.num_alphas

#========================================================================================
#models / bases
class FFXModel:
    def __init__(self, coefs_n, bases_n, coefs_d, bases_d, varnames=None):
        """
        @arguments
          coefs_n -- 1d array of float -- coefficients for numerator.
          bases_n -- list of *Base -- bases for numerator
          coefs_d -- 1d array of float -- coefficients for denominator
          bases_d -- list of *Base -- bases for denominator
          varnames -- list of string
        """
        #preconditions
        assert 1+len(bases_n) == len(coefs_n)  #offset + numer_bases == numer_coefs
        assert   len(bases_d) == len(coefs_d)  #denom_bases == denom_coefs

        #make sure that the coefs line up with their 'pretty' versions
        coefs_n = numpy.array([float(coefStr(coef)) for coef in coefs_n])
        coefs_d = numpy.array([float(coefStr(coef)) for coef in coefs_d])

        #reorder numerator bases from highest-to-lowest influence 
        # -but keep offset 0th of course
        offset = coefs_n[0]
        coefs_n2 = coefs_n[1:]
        I = numpy.argsort(numpy.abs(coefs_n2))[::-1]
        coefs_n = [offset] + [coefs_n2[i] for i in I]
        bases_n = [bases_n[i] for i in I]

        #reorder denominator bases from highest-to-lowest influence
        I = numpy.argsort(numpy.abs(coefs_d))[::-1]
        coefs_d = [coefs_d[i] for i in I]
        bases_d = [bases_d[i] for i in I]

        #store values
        self.varnames = varnames
        self.coefs_n = coefs_n
        self.bases_n = bases_n
        self.coefs_d = coefs_d
        self.bases_d = bases_d

    def numBases(self):
        """Return total number of bases"""
        return len(self.bases_n) + len(self.bases_d)
        
    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        N = X.shape[0]
        
        #numerator
        y = numpy.zeros(N, dtype=float)
        y += self.coefs_n[0]
        for (coef, base) in itertools.izip(self.coefs_n[1:], self.bases_n):
            y += coef * base.simulate(X)

        #denominator
        if self.bases_d:
            denom_y = numpy.zeros(N, dtype=float)
            denom_y += 1.0
            for (coef, base) in itertools.izip(self.coefs_d, self.bases_d):
                denom_y += coef * base.simulate(X)
            y /= denom_y
            
        return y

    def __str__(self):
        return self.str2()

    def str2(self, maxlen=100000):
        include_denom = bool(self.bases_d)

        s = ''
        #numerator
        if include_denom and len(self.coefs_n)>1: s += '('
        numer_s = ['%s' % coefStr(self.coefs_n[0])]
        for (coef, base) in itertools.izip(self.coefs_n[1:], self.bases_n):
            numer_s += ['%s*%s' % (coefStr(coef), base)]
        s += ' + '.join(numer_s)
        if include_denom and len(self.coefs_n)>1: s += ')'

        #denominator
        if self.bases_d:
            s += ' / ('
            denom_s = ['1.0']
            for (coef, base) in itertools.izip(self.coefs_d, self.bases_d):
                denom_s += ['%s*%s' % (coefStr(coef), base)]
            s += ' + '.join(denom_s)
            s += ')'
            
        #change xi to actual variable names
        for var_i in xrange(len(self.varnames)-1, -1, -1):
            s = s.replace('x%d' % var_i, self.varnames[var_i])
        s = s.replace('+ -', '- ')
            
        #truncate long strings
        if len(s) > maxlen:
            s = s[:maxlen] + '...' 

        return s

class SimpleBase:
    """e.g. x4^2"""
    def __init__(self, var, exponent):
        self.var = var
        self.exponent = exponent

    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        return X[:,self.var] ** self.exponent

    def __str__(self):
        if self.exponent == 1:
            return 'x%d' % self.var
        else:
            return 'x%d^%g' % (self.var, self.exponent)
                                
class OperatorBase:
    """e.g. log(x4^2)"""
    def __init__(self, simple_base, nonlin_op, thr=INF):
        """
        @arguments
          simple_base -- SimpleBase
          nonlin_op -- one of OPS
          thr -- None or float -- depends on nonlin_op
        """
        self.simple_base = simple_base
        self.nonlin_op = nonlin_op
        self.thr = thr

    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        op = self.nonlin_op
        ok = True
        y_lin = self.simple_base.simulate(X)

        if op == OP_ABS:     ya = abs(y_lin)
        elif op == OP_MAX0:  ya = numpy.clip(y_lin, 0.0, INF)
        elif op == OP_MIN0:  ya = numpy.clip(y_lin, -INF, 0.0)
        elif op == OP_LOG10:
            #safeguard against: log() on values <= 0.0
            mn, mx = min(y_lin), max(y_lin)
            if mn <= 0.0 or scipy.isnan(mn) or mx == INF or scipy.isnan(mx):
                ok = False
            else:
                ya = numpy.log10(y_lin)
        elif op == OP_GTH:   ya = numpy.clip(self.thr - y_lin, 0.0, INF)
        elif op == OP_LTH:   ya = numpy.clip(y_lin - self.thr, 0.0, INF)
        else:                raise 'Unknown op %d' % op

        if ok: #could always do ** exp, but faster ways if exp is 0,1
            y = ya
        else:
            y = INF * numpy.ones(X.shape[0], dtype=float)    
        return y
    
    def __str__(self):
        op = self.nonlin_op
        simple_s = str(self.simple_base)
        if op == OP_ABS:     return 'abs(%s)' % simple_s
        elif op == OP_MAX0:  return 'max(0, %s)' % simple_s
        elif op == OP_MIN0:  return 'min(0, %s)' % simple_s
        elif op == OP_LOG10: return 'log10(%s)' % simple_s
        elif op == OP_GTH:   return ('max(0,%s-%s)' % (simple_s, coefStr(self.thr))).replace('--','+')
        elif op == OP_LTH:   return 'max(0,%s-%s)' % (coefStr(self.thr), simple_s)
        else:                raise 'Unknown op %d' % op

class ProductBase:
    """e.g. x2^2 * log(x1^3)"""
    def __init__(self, base1, base2):
        self.base1 = base1
        self.base2 = base2

    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        yhat1 = self.base1.simulate(X)
        yhat2 = self.base2.simulate(X)
        return yhat1 * yhat2

    def __str__(self):
        return '%s * %s' % (self.base1, self.base2)

class ConstantModel:
    """e.g. 3.2"""
    def __init__(self, constant, numvars):
        """
        @description        
            Constructor.
    
        @arguments        
            constant -- float -- constant value returned by this model
            numvars -- int -- number of input variables to this model
        """ 
        self.constant = float(constant) 
        self.numvars = numvars

    def numBases(self):
        """Return total number of bases"""
        return 0

    def simulate(self, X):
        """
        @arguments
          X -- 2d array of [sample_i][var_i] : float
        @return
          y -- 1d array of [sample_i] : float
        """
        N = X.shape[0]
        if scipy.isnan(self.constant): #corner case
            yhat = numpy.array([INF] * N)
        else: #typical case
            yhat = numpy.ones(N, dtype=float) * self.constant  
        return yhat
    
    def __str__(self):
        return self.str2()

    def str2(self, dummy_arg=None):
        return coefStr(self.constant)



#==============================================================================
#Model factories

class MultiFFXModelFactory:

    def build(self, train_X, train_y, test_X, test_y, varnames=None):
        """
        @description
          Builds FFX models at many different settings, then merges the results
          into a single Pareto Optimal Set.

        @argument
          train_X -- 2d array of [sample_i][var_i] : float -- training inputs 
          test_y -- 1d array of [sample_i] : float -- training outputs
          test_X -- 2d array -- testing inputs
          test_y -- 1d array -- testing outputs
          varnames -- list of string -- variable names

        @return
          models -- list of FFXModel -- Pareto-optimal set of models
        """
        
        if isinstance(train_X, pandas.DataFrame):
            varnames = train_X.columns
            train_X = train_X.as_matrix()
            test_X = test_X.as_matrix()
        if isinstance(train_X, numpy.ndarray) and varnames == None:
            raise Exception, 'varnames required for numpy.ndarray'
            
        print 'Build(): begin. {2} variables, {1} training samples, {0} test samples'.format(test_X.shape[1], *train_X.shape)
        
        models = []
        min_y = min(min(train_y), min(test_y))
        max_y = max(max(train_y), max(test_y))

        #build all combinations of approaches, except for (a) features we don't consider
        # and (b) too many features at once
        approaches = []
        print "Approaches:"
        if CONSIDER_INTER: inters = [1] #inter=0 is a subset of inter=1
        else:              inters = [0]
        for inter in inters: 
            for denom in [0,1]:
                if denom==1 and not CONSIDER_DENOM: continue
                for expon in [0,1]:
                    if expon==1 and not CONSIDER_EXPON: continue
                    if expon==1 and inter==1: continue #never need both exponent and inter
                    for nonlin in [0,1]:
                        if nonlin==1 and not CONSIDER_NONLIN: continue
                        for thresh in [0,1]:
                            if thresh==1 and not CONSIDER_THRESH: continue
                            approach = [inter, denom, expon, nonlin, thresh]
                            if sum(approach) >= 4: continue #not too many features at once
                            approaches.append(approach)
                            print "  ", _approachStr(approach)

        for (i, approach) in enumerate(approaches): 
            print '-' * 200
            print 'Build with approach %d/%d (%s): begin' % \
                (i+1, len(approaches), _approachStr(approach))
            ss = FFXBuildStrategy(approach)

            next_models = FFXModelFactory().build(train_X, train_y, ss, varnames)

            #set test_nmse on each model
            for model in next_models:
                test_yhat = model.simulate(test_X)
                model.test_nmse = nmse(test_yhat, test_y, min_y, max_y)

            #pareto filter
            print '  STEP 3: Nondominated filter'
            next_models = self._nondominatedModels(next_models)
            models += next_models
            print 'Build with approach %d/%d (%s): done.  %d model(s).' % \
                (i+1, len(approaches), _approachStr(approach), len(next_models))
            print 'Models:'
            for model in next_models:
                print "num_bases=%d, test_nmse=%.6f, model=%s" % \
                    (model.numBases(), model.test_nmse, model.str2(500))

        #final pareto filter
        models = self._nondominatedModels(models)

        #log nondominated models
        print '=' * 200
        print '%d nondominated models (wrt test error & num. bases):' % len(models)
        for (i, model) in enumerate(models):
            print "Nondom model %d/%d: num_bases=%d, test_nmse=%.6f, model=%s" % \
                (i+1, len(models), model.numBases(), model.test_nmse, model.str2(500))

        return models

    def _FFXapproach(self, inter, denom, expon, nonlin, thresh):
        return 'FFX inter%d denom%d expon%d nonlin%d thresh%d' % \
            (inter, denom, expon, nonlin, thresh)

    def _nondominatedModels(self, models):
        test_nmses = [model.test_nmse for model in models]
        num_bases = [model.numBases() for model in models]
        I = nondominatedIndices2d(test_nmses, num_bases)
        models = [models[i] for i in I]

        I = numpy.argsort([model.numBases() for model in models])
        models = [models[i] for i in I]

        return models


class FFXModelFactory:

    def build(self, X, y, ss, varnames=None):
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
        if isinstance(X, pandas.DataFrame):
            varnames = X.columns
            X = X.as_matrix()
        if isinstance(X, numpy.ndarray) and varnames == None:
            raise Exception, 'varnames required for numpy.ndarray'
            
        if X.ndim == 1:
            self.nrow, self.ncol = len(X), 1
        elif X.ndim == 2:
            self.nrow, self.ncol = X.shape
        else:
            raise Exception, 'X is wrong dimensionality.'        
        
        y = numpy.asarray(y)
        if self.nrow != len(y):
            raise Exception, 'X sample count and y sample count do not match'
        
        if self.ncol == 0:
            print '  Corner case: no input vars, so return a ConstantModel'
            return [ConstantModel(y.mean(), 0)]
        
        #Main work... 
        
        #build up each combination of all {var_i} x {op_j}, except for
        # when a combination is unsuccessful
        print '  STEP 1A: Build order-1 bases: begin'
        order1_bases = []
        for var_i in range(self.ncol):
            #if (var_i+1) % 10 == 0: print '    Build bases at var %d/%d' % (var_i+1, X.shape[0])
            for exponent in ss.exprExponents():
                if exponent == 0.0: continue

                #'lin' version of base
                simple_base = SimpleBase(var_i, exponent)
                lin_yhat = simple_base.simulate(X)
                if exponent in [1.0, 2.0] or not yIsPoor(lin_yhat): #checking exponent is a speedup
                    order1_bases.append(simple_base) 
                    
                    #add e.g. OP_ABS, OP_MAX0, OP_MIN0, OP_LOG10
                    for nonlin_op in ss.nonlinOps(): 
                        #ignore cases where op has no effect
                        if nonlin_op == OP_ABS and exponent in [-2, +2]: continue 
                        if nonlin_op == OP_MAX0 and min(lin_yhat) >= 0: continue 
                        if nonlin_op == OP_MIN0 and max(lin_yhat) <= 0: continue 

                        nonsimple_base = OperatorBase(simple_base, nonlin_op, None)
                        nonsimple_base.var = var_i #easy access when considering interactions

                        nonlin_yhat = nonsimple_base.simulate(X)
                        if numpy.all(nonlin_yhat == lin_yhat): continue #op has no effect
                        if not yIsPoor(nonlin_yhat):
                            order1_bases.append(nonsimple_base)

                    #add e.g. OP_GTH, OP_LTH
                    if exponent == 1.0 and ss.thresholdOps():
                        minx, maxx = min(X[:,var_i]), max(X[:,var_i])
                        rangex = maxx - minx
                        stepx = 0.8 * rangex / float(ss.num_thrs_per_var+1)
                        thrs = numpy.arange(
                            minx + 0.2*rangex, maxx - 0.2*rangex + 0.1*rangex, stepx)
                        for threshold_op in ss.thresholdOps(): 
                            for thr in thrs:
                                nonsimple_base = OperatorBase(simple_base, threshold_op, thr)
                                nonsimple_base.var = var_i #easy access when considering interactions
                                order1_bases.append(nonsimple_base)

        print '  STEP 1A: Build order-1 bases: done.  Have %d order-1 bases.' % len(order1_bases)
        #print '  The order-1 bases: %s' % basesStr(order1_bases)

        var1_models = None
        if ss.includeInteractions():
            #find base-1 influences
            print '  STEP 1B: Find order-1 base infls: begin'
            max_num_bases = len(order1_bases) #no limit
            target_train_nmse = 0.01
            models = self._basesToModels(
                ss, varnames, order1_bases, X, y, max_num_bases, target_train_nmse) 
            if models is None: #fit failed.
                model = ConstantModel(y[0], 0)
                return [model]
            var1_models = models

            model = models[-1] #use most-explaining model (which also has the max num bases)
            order1_bases = model.bases_n + model.bases_d

            #order bases by influence
            order1_infls = numpy.abs(list(model.coefs_n[1:]) + list(model.coefs_d)) #influences
            I = numpy.argsort(-1 * order1_infls)
            order1_bases = [order1_bases[i] for i in I] 
            print '  STEP 1B: Find order-1 base infls: done'

            #don't let inter coeffs swamp linear ones; but handle more when n small
            n_order1_bases = len(order1_bases)
            max_n_order2_bases = 3 * math.sqrt(n_order1_bases)  #default
            max_n_order2_bases = max(max_n_order2_bases, 10)                 #lower limit
            max_n_order2_bases = max(max_n_order2_bases, 2 * n_order1_bases) #  ""
            if ss.includeDenominator(): max_n_order2_bases = min(max_n_order2_bases, 4000) #upper limit
            else:                       max_n_order2_bases = min(max_n_order2_bases, 8000) # ""
            
            #build up order2 bases
            print '  STEP 1C: Build order-2 bases: begin'

            #  -always have all xi*xi terms
            order2_bases = []
            order2_basetups = set() # set of (basei_id, basej_id) tuples
            for i, basei in enumerate(order1_bases):
                if basei.__class__ != SimpleBase: continue #just xi
                if basei.exponent != 1.0: continue #just exponent==1

                order2_exponent = 2
                combined_base = SimpleBase(var_i, 2)
                order2_bases.append(combined_base)

                tup = (id(basei), id(basei))
                order2_basetups.add(tup)

            # -then add other terms
            for max_base_i in xrange(len(order1_bases)):
                for i in xrange(max_base_i):
                    basei = order1_bases[i]
                    for j in xrange(max_base_i):
                        if j >= i: continue #disallow mirror image
                        basej = order1_bases[j]
                        tup = (id(basei), id(basej))
                        if tup in order2_basetups: continue #no duplicate pairs
                        combined_base = ProductBase(basei, basej)
                        order2_bases.append(combined_base)
                        order2_basetups.add(tup)

                        if len(order2_bases) >= max_n_order2_bases: break #for j
                    if len(order2_bases) >= max_n_order2_bases: break #for i
                if len(order2_bases) >= max_n_order2_bases: break #for max_base_i

            print '  STEP 1C: Build order-2 bases: done.  Have %d order-2 bases.' % len(order2_bases)
            #print '  Some order-2 bases: %s' % basesStr(order2_bases[:10])
            bases = order1_bases + order2_bases
        else:
            bases = order1_bases

        #all bases. Stop based on target nmse, not number of bases
        print '  STEP 2: Regress on all %d bases: begin.' % len(bases)
        var2_models = self._basesToModels(
            ss, varnames, bases, X, y, ss.final_max_num_bases, ss.final_target_train_nmse)
        print '  STEP 2: Regress on all %d bases: done.' % len(bases)

        #combine models having 1-var with models having 2-vars
        if var1_models is None and var2_models is None:
            models = []
        elif var1_models is None and var2_models is not None:
            models = var2_models
        elif var1_models is not None and var2_models is None: 
            models = var1_models
        else: #var1_models is not None and var2_models is not None
            models = var1_models + var2_models

        #add constant; done
        models = [ConstantModel(numpy.mean(y), X.shape[0])] + models
        return models

    def _basesToModels(self, ss, varnames, bases, X, y, max_num_bases, target_train_nmse):
        #compute regress_X
        if ss.includeDenominator(): regress_X = numpy.zeros((self.nrow, len(bases)*2), dtype=float)
        else:                       regress_X = numpy.zeros((self.nrow, len(bases)),   dtype=float)
        for i, base in enumerate(bases):
            base_y = base.simulate(X)
            regress_X[:,i] = base_y #numerators
            if ss.includeDenominator():
                regress_X[:,len(bases)+i] = -1.0 * base_y * y #denominators
        
        #compute models.  
        models = self._pathwiseLearn(ss, varnames, bases, X, regress_X, y, 
                                     max_num_bases, target_train_nmse)
        return models
                    
    def _pathwiseLearn(self, ss, varnames, bases, X_orig, X_orig_regress, y_orig,
                       max_num_bases, target_nmse, **fit_params):
        """Adapted from enet_path() in scikits.learn.linear_model.
        http://scikit-learn.sourceforge.net/modules/linear_model.html
        Compute Elastic-Net path with coordinate descent.  
        Returns list of model (or None if failure)."""
        
        print '    Pathwise learn: begin. max_num_bases=%d' % max_num_bases
        max_iter = 1000 #default 1000. magic number.
        
        #Condition X and y: 
        # -"unbias" = rescale so that (mean=0, stddev=1) -- subtract each row's mean, then divide by stddev
        # -X transposed
        # -X as fortran array
        (X_unbiased, y_unbiased, X_avgs, X_stds, y_avg, y_std) = self._unbiasedXy(X_orig_regress, y_orig)
        X_unbiased = numpy.asfortranarray(X_unbiased) # make data contiguous in memory

        n_samples = X_unbiased.shape[0]
        vals = numpy.dot(X_unbiased.T, y_unbiased)
        vals = [val for val in vals if not scipy.isnan(val)]
        if vals: alpha_max = numpy.abs(max(vals) / (n_samples * ss.rho()))
        else:    alpha_max = 1.0 #backup: pick a value from the air

        #alphas = lotsa alphas at beginning, and usual rate for rest
        st, fin = numpy.log10(alpha_max*ss.eps()), numpy.log10(alpha_max)
        alphas1 = numpy.logspace(st, fin, num=ss.numAlphas()*10)[::-1][:ss.numAlphas()/4]
        alphas2 = numpy.logspace(st, fin, num=ss.numAlphas())
        alphas = sorted(set(alphas1).union(alphas2), reverse=True)

        if not 'precompute' in fit_params or fit_params['precompute'] is True:
            fit_params['precompute'] = numpy.dot(X_unbiased.T, X_unbiased)
            if not 'Xy' in fit_params or fit_params['Xy'] is None:
                fit_params['Xy'] = numpy.dot(X_unbiased.T, y_unbiased)

        models = [] #build this up
        nmses = [] #for detecting stagnation
        cur_unbiased_coefs = None # init values for coefs
        start_time = time.time()
        for (alpha_i, alpha) in enumerate(alphas):
            #compute (unbiased) coefficients. Recall that mean=0 so no intercept needed
            clf = ElasticNetWithTimeout(alpha=alpha, rho=ss.rho(), fit_intercept=False)
            try:
                clf.fit(X_unbiased, y_unbiased, coef_init=cur_unbiased_coefs, 
                        max_iter=max_iter, **fit_params)
            except TimeoutError:
                print '    Regularized update failed. Returning None'
                return None #failure
            cur_unbiased_coefs = clf.coef_.copy() 

            #compute model; update models
            #  -"rebias" means convert from (mean=0, stddev=1) to original (mean, stddev)
            coefs = self._rebiasCoefs([0.0] + list(cur_unbiased_coefs), X_stds, X_avgs, y_std, y_avg)
            (coefs_n, bases_n, coefs_d, bases_d) = self._allocateToNumerDenom(ss, bases, coefs)
            model = FFXModel(coefs_n, bases_n, coefs_d, bases_d, varnames)
            models.append(model)

            #update nmses
            nmse_unbiased = nmse(numpy.dot(cur_unbiased_coefs, X_unbiased.T), y_unbiased,
                                 min(y_unbiased), max(y_unbiased))
            nmses.append(nmse_unbiased)

            #log
            num_bases = len(numpy.nonzero(cur_unbiased_coefs)[0])
            if (alpha_i==0) or (alpha_i+1) % 50 == 0:
                print '      alpha %d/%d (%3e): num_bases=%d, nmse=%.6f, time %.2f s' % \
                    (alpha_i+1, len(alphas), alpha, num_bases, nmse_unbiased, time.time() - start_time)

            #maybe stop
            if scipy.isinf(nmses[-1]):
                print '    Pathwise learn: Early stop because nmse is inf'
                return None
            if nmse_unbiased < target_nmse:
                print '    Pathwise learn: Early stop because nmse < target'
                return models
            if num_bases > max_num_bases:
                print '    Pathwise learn: Early stop because num bases > %d' % max_num_bases
                return models
            if len(nmses) > 15 and round(nmses[-1], 4) == round(nmses[-15], 4):
                print '    Pathwise learn: Early stop because nmses stagnated'
                return models

        print '    Pathwise learn: done'
        return models

    def _allocateToNumerDenom(self, ss, bases, coefs):
        """Prune out nonzero coefficients/bases.  Allocate to numerator vs. denominator."""
        if ss.includeDenominator():
            assert 1+len(bases)+len(bases) == len(coefs) #offset + numer_bases + denom_bases
            n_bases = len(bases)
            coefs_n = [coefs[0]] + [coef for coef in coefs[1:n_bases+1] if coef != 0]
            bases_n = [base for (base, coef) in itertools.izip(bases, coefs[1:n_bases+1]) if coef != 0]
            coefs_d = [coef for coef in coefs[n_bases+1:] if coef != 0]
            bases_d = [base for (base, coef) in itertools.izip(bases, coefs[n_bases+1:]) if coef != 0]

        else:
            assert 1+len(bases) == len(coefs) #offset + numer_bases + denom_bases
            coefs_n = [coefs[0]] + [coef for coef in coefs[1:] if coef != 0]
            bases_n = [base for (base, coef) in itertools.izip(bases, coefs[1:]) if coef != 0]
            coefs_d = []
            bases_d = []

        return (coefs_n, bases_n, coefs_d, bases_d)

    def _unbiasedXy(self, Xin, yin):
        """Make all input rows of X, and y, to have mean=0 stddev=1 """ 
        #unbiased X
        X_avgs, X_stds = Xin.mean(0), Xin.std(0)
        X_unbiased = (Xin - X_avgs) / X_stds
        
        #unbiased y
        y_avg, y_std = yin.mean(0), yin.std(0)
        y_unbiased = (yin - y_avg) / y_std
        
        return (X_unbiased, y_unbiased, X_avgs, X_stds, y_avg, y_std)

    def _rebiasCoefs(self, unbiased_coefs, X_stds, X_avgs, y_std, y_avg):
        """Given the coefficients that were learned using unbiased training data, rebias the
        coefficients so that they are at the scale of the real training X and y."""
        #preconditions
        assert unbiased_coefs is not None
        assert len(unbiased_coefs) == (len(X_stds)+1) == (len(X_avgs)+1), \
            (len(unbiased_coefs), (len(X_stds)+1), (len(X_avgs)+1))

        #main work
        n = len(X_stds)
        coefs = numpy.zeros(n+1, dtype=float)
        coefs[0] = unbiased_coefs[0] * y_std + y_avg
        for j in range(1,n+1):
            coefs[j] = unbiased_coefs[j] * y_std / X_stds[j-1]
            coefs[0] -= coefs[j] * X_avgs[j-1]
        return coefs


#========================================================================================
#Revise linear_model.coordinate_descent.ElasticNet.fit() to handle when it hangs
#http://www.saltycrane.com/blog/2010/04/using-python-timeout-decorator-uploading-s3/

class TimeoutError(Exception):
    def __init__(self, value = "Timed Out"):
        self.value = value
    def __str__(self):
        return repr(self.value)

def timeout(seconds_before_timeout):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds_before_timeout)
            try:
                result = f(*args, **kwargs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result
        new_f.func_name = f.func_name
        return new_f
    return decorate

class ElasticNetWithTimeout(ElasticNet):
    @timeout(MAX_TIME_REGULARIZE_UPDATE) #if this freezes, then exit with a TimeoutError
    def fit(self, *args, **kwargs):
        return ElasticNet.fit(self, *args, **kwargs)

#========================================================================================
#utility classes / functions
def nondominatedIndices2d(cost0s, cost1s):
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
    cost0s, cost1s = numpy.asarray(cost0s), numpy.asarray(cost1s)
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

def nmse(yhat, y, min_y, max_y):
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
    """
    #base case: no entries
    if len(yhat) == 0:
        return 0.0

    #base case: both yhat and y are constant, and same values
    if (max_y == min_y) and (max(yhat) == min(yhat) == max(y) == min(y)):
        return 0.0

    #main case
    assert max_y > min_y, 'max_y=%g was not > min_y=%g' % (max_y, min_y)
    yhat_a, y_a = numpy.asarray(yhat), numpy.asarray(y)
    y_range = float(max_y - min_y)
    try:
        result = math.sqrt(numpy.mean(((yhat_a - y_a) / y_range) ** 2))
        if scipy.isnan(result):
            return INF
        return result
    except:
        return INF

def yIsPoor(y):
    """Returns True if y is not usable"""
    return max(scipy.isinf(y)) or max(scipy.isnan(y))


def coefStr(x):
    """Gracefully print a number to 3 significant digits.  See _testCoefStr in unit tests"""
    if x == 0.0:        s = '0'
    elif abs(x) < 1e-4: s = ('%.2e' % x).replace('e-0', 'e-')
    elif abs(x) < 1e-3: s = '%.6f' % x
    elif abs(x) < 1e-2: s = '%.5f' % x
    elif abs(x) < 1e-1: s = '%.4f' % x
    elif abs(x) < 1e0:  s = '%.3f' % x
    elif abs(x) < 1e1:  s = '%.2f' % x
    elif abs(x) < 1e2:  s = '%.1f' % x
    elif abs(x) < 1e4:  s = '%.0f' % x
    else:               s = ('%.2e' % x).replace('e+0', 'e')
    return s

def basesStr(bases):
    """Pretty print list of bases"""
    return ', '.join([str(base) for base in bases])

def rail(x, minx, maxx):
    return max(minx, max(maxx, x))
