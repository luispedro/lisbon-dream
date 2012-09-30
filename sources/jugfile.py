from jug import Task, TaskGenerator
from leave1out import leave1out
from projection import random_project
from randomlearner import random_learner
from selectlearner import *
from regularized import ridge_regression, lasso_regression, lasso_regression_with_learning
from regularized import lasso_relaxed_after_learning
from regularized import lasso_relaxed_after_learning, lasso_relaxed
from regularized import *
from preproc import *
import numpy as np
from milk.supervised.classifier import ctransforms
from milk.supervised.normalise import zscore_normalise

leave1out = TaskGenerator(leave1out)
@TaskGenerator
def print_results(results):
    with open('results.txt', 'w') as output:
        for name in sorted(results.keys(), key=lambda k: np.mean(results[k][0])):
            val = results[name]
            val = np.mean(val[0])
            print >>output, '{0:<64}: {1: .2%}'.format(name, val)


@TaskGenerator
def prune_features(features_labels, frac):
    features, labels = features_labels
    return prune_similar(features, frac=frac), labels

@TaskGenerator
def thresh(features_labels):
    features, labels = features_labels
    mu = features.mean(0)
    std = features.std(0)
    features = (features > mu + 2*std) | (features < mu - 2*std)
    return features, labels


class add_model(supervised_model):
    def __init__(self, base, b):
        self.base = base
        self.b = b

    def apply(self, features):
        return self.base.apply(features) + self.b

class norm_learner(object):
    def __init__(self, base, axis):
        self.base = base
        self.axis = axis

    def train(self, features, labels):
        from milk.unsupervised import center
        labels, mean = center(labels, axis=self.axis)
        return self.base.train(features, labels)

class znorm_learner(object):
    def __init__(self, base, axis):
        self.base = base
        self.axis = axis

    def train(self, features, labels):
        from milk.unsupervised import zscore
        labels = zscore(labels, axis=self.axis)
        return self.base.train(features, labels)


rna_ge_gosweigths_add = Task(rna_ge_gosweigths, 'add')
rna_ge_gosweigths_maxabs = Task(rna_ge_gosweigths, 'maxabs')


results = {}
for lname,data in [
                ('rna+ge+act+zs', Task(ge_rna_valid)),
                ('rna+ge+act(ma)+zs', Task(ge_rna_valid,'maxabs')),
                ('gow', rna_ge_gosweigths_add),
                ('gow-thresh', thresh(rna_ge_gosweigths_add)),
                ('gow-ma', rna_ge_gosweigths_maxabs),
                ('gow-ma-thresh', thresh(rna_ge_gosweigths_maxabs)),
                ('prune(gow, .5)', prune_features(rna_ge_gosweigths_add, .5)),
                ('prune(gow-ma, .5)', prune_features(rna_ge_gosweigths_maxabs, .5)),
                ('prune(gow, .9)', prune_features(rna_ge_gosweigths_add, .9)),
                ('prune(gow-ma, .9)', prune_features(rna_ge_gosweigths_maxabs, .9)),
                ('prune(gow, .95)', prune_features(rna_ge_gosweigths_add, .95)),
                ('prune(gow-ma, .95)', prune_features(rna_ge_gosweigths_maxabs, .95)),
                ]:
    features = data[0]
    labels = data[1]

    for name,learner in [
            #('relaxed', lasso_relaxed_after_learning()),
            ('relaxed', lasso_relaxed(.000225010113525, .1)),
            ('sel-relaxed', ctransforms(remove_constant_features(), select_learner(12), lasso_relaxed(.000225010113525, .1))),
            ('zs-sel-relaxed', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_relaxed(.000225010113525, .1))),
            ('random(1)', random_learner(1)),
            ('random(2)', random_learner(2)),
            ('random(3)', random_learner(3)),
            ('ridge(.128)', ridge_regression(.128)),
            ('ridge(.001)', ridge_regression(.001)),
            ('ridge(1e-6)', ridge_regression(1e-6)),
            ('select+ridge(.001)', ctransforms(remove_constant_features(), select_learner(12), ridge_regression(.001))),
            ('ridge(1.)', ridge_regression(1.)),
            ('sel+lasso(1e-7)', ctransforms(remove_constant_features(), select_learner(12), lasso_regression(1e-7))),
            ('sel+lasso(1e-5)', ctransforms(remove_constant_features(), select_learner(16), lasso_regression(1e-5))),
            ('sel+lasso(0.000225010113525)', ctransforms(remove_constant_features(), select_learner(16), lasso_regression(0.000225010113525))),
            ('zs+sel+lasso(0.000225010113525)', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(16), lasso_regression(0.000225010113525))),
            ('zs+sel+lasso_learn', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_regression_with_learning())),
            ('zs+sel+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_regression_guess())),
            ('zs+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), lasso_regression_guess())),
            ('zs+sel+min0+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), min_to_zero(), lasso_regression_guess())),
            ('zs+min0+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), min_to_zero(), lasso_regression_guess())),
            ('min0+lasso_guess', ctransforms(remove_constant_features(), min_to_zero(), lasso_regression_guess())),
            ('rproject(12)', random_project(12)),
            ('rproject(24)', random_project(24)),
            ('rproject(128, ridge(.01))', random_project(128, ridge_regression(.01))),
            ('rproject(1024, lasso_learn)', random_project(1024, lasso_regression_with_learning())),
            ('rproject(1024, lasso_guess)', random_project(1024, lasso_regression_guess())),
            ('rproject(8192, lasso_guess)', random_project(8192, lasso_regression_guess())),
            ]:
        results[lname,name] = leave1out(learner, features, labels)
        learner0 = norm_learner(learner, 0)
        learner1 = norm_learner(learner, 1)
        results['{0}-normed0'.format(lname),name] = leave1out(learner0, features, labels)
        results['{0}-normed1'.format(lname),name] = leave1out(learner1, features, labels)
        zlearner0 = znorm_learner(learner, 0)
        zlearner1 = znorm_learner(learner, 1)
        results['{0}-znormed0'.format(lname),name] = leave1out(zlearner0, features, labels)
        results['{0}-znormed1'.format(lname),name] = leave1out(zlearner1, features, labels)

print_results(results)
