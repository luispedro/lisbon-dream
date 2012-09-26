from jug import Task, TaskGenerator
from leave1out import leave1out
from projection import random_project
from randomlearner import random_learner
from selectlearner import *
from regularized import ridge_regression, lasso_regression, lasso_path_regression
from preproc import *
import numpy as np
from milk.supervised.classifier import ctransforms

leave1out = TaskGenerator(leave1out)
@TaskGenerator
def print_results(results):
    with open('results.txt', 'w') as output:
        for name in sorted(results.keys(), key=lambda k: np.mean(results[k][0])):
            val = results[name]
            val = np.mean(val[0])
            print >>output, '{0:<64}: {1: .2%}'.format(name, val)



def rna_ge_gosweigths_pruned(frac=.5):
    features, labels = rna_ge_gosweigths()
    return prune_similar(features, frac=.5), labels

class norm_learner(object):
    def __init__(self, base, axis):
        self.base = base
        self.axis = axis

    def train(self, features, labels):
        return self.base.train(features, normlabels(labels, self.axis))

results = {}
for lname,data in [
                ('rna+ge', Task(rna_ge_concatenated)),
                ('active', Task(rna_seq_active_only)),
                ('rna+ge+active+zscore', Task(ge_rna_valid)),
                ('rna+ge+active(maxabs)+zscore', Task(ge_rna_valid,'maxabs')),
                ('gosweigths', Task(rna_ge_gosweigths)),
                ('gosweigths-maxabs', Task(rna_ge_gosweigths, 'maxabs')),
                ('prune(gosweigths, .5)', Task(rna_ge_gosweigths_pruned)),
                ('prune(gosweigths, .9)', Task(rna_ge_gosweigths_pruned, .9)),
                ('prune(gosweigths, .95)', Task(rna_ge_gosweigths_pruned, .95)),
                ]:
    features = data[0]
    labels = data[1]

    for name,learner in [
            ('ridge(.128)', ridge_regression(.128)),
            ('ridge(.001)', ridge_regression(.001)),
            ('select+ridge(.001)', ctransforms(remove_constant_features(), select_learner(), ridge_regression(.001))),
            ('select+lasso_path', ctransforms(remove_constant_features(), select_learner(), lasso_path_regression())),
            ('ridge(1.)', ridge_regression(1.)),
            ('lasso_path', lasso_path_regression()),
            ('random(1)', random_learner(1)),
            ('random(2)', random_learner(2)),
            ('random(3)', random_learner(3)),
            ('rproject(12)', random_project(12)),
            ('rproject(24)', random_project(24)),
            ('rproject(128, ridge(.01))', random_project(128, ridge_regression(.01))),
            ('rproject(128, lasso_path)', random_project(128, lasso_path_regression())),
            ('rproject(256, lasso_path)', random_project(256, lasso_path_regression())),
            ('rproject(1024, lasso_path)', random_project(1024, lasso_path_regression())),
            ]:
        results['{0}-{1}'.format(lname,name)] = leave1out(learner, features, labels)
        learner0 = norm_learner(learner, 0)
        learner1 = norm_learner(learner, 1)
        results['{0}-normed0-{1}'.format(lname,name)] = leave1out(learner0, features, labels)
        results['{0}-normed1-{1}'.format(lname,name)] = leave1out(learner1, features, labels)

print_results(results)
