from jug import Task, TaskGenerator
from leave1out import leave1out
from projection import random_project
from randomlearner import random_learner
from regularized import ridge_regression, lasso_regression
from preproc import *
import numpy as np

leave1out = TaskGenerator(leave1out)
@TaskGenerator
def print_results(results):
    with open('results.txt', 'w') as output:
        for name in sorted(results.keys(), key=lambda k: np.mean(results[k][0])):
            val = results[name]
            val = np.mean(val[0])
            print >>output, '{0:<64}: {1: .2%}'.format(name, val)


def zscore_rna_seq():
    from milk.unsupervised import zscore
    features, labels = rna_seq_active_only()
    return zscore(features, axis=1), labels

def rna_ge_gosweigths_pruned():
    features, labels = rna_ge_gosweigths()
    return prune_similar(features, frac=.5), labels

class norm_learner(object):
    def __init__(self, base):
        self.base = base

    def train(self, features, labels):
        return self.base.train(features, normlabels.f(labels, 1))

normlabels = TaskGenerator(normlabels)

results = {}
for lname,data in [
                ('rna+ge', Task(rna_ge_concatenated)),
                ('active', Task(rna_seq_active_only)),
                ('active+zscore', Task(zscore_rna_seq)),
                ('rna+ge+active+zscore', Task(ge_rna_valid)),
                ('rna+ge+active(maxabs)+zscore', Task(ge_rna_valid,'max(abs)')),
                ('gosweigths', Task(rna_ge_gosweigths)),
                ('gosweigths-maxabs', Task(rna_ge_gosweigths, 'maxabs')),
                ('prune(gosweigths, .5)', Task(rna_ge_gosweigths_pruned)),
                ]:
    features = data[0]
    labels = data[1]

    for name,learner in [
            ('ridge(.128)', ridge_regression(.128)),
            ('ridge(.001)', ridge_regression(.001)),
            ('ridge(1.)', ridge_regression(1.)),
            ('lasso(.000000002)', lasso_regression(.000000002)),
            ('lasso(.0002)', lasso_regression(.0002)),
            ('lasso(.05)', lasso_regression(.05)),
            ('lasso(.5)', lasso_regression(.5)),
            ('lasso(.8)', lasso_regression(.8)),
            ('lasso(2)', lasso_regression(2)),
            ('random(1)', random_learner(1)),
            ('random(2)', random_learner(2)),
            ('random(3)', random_learner(3)),
            ('rproject(12)', random_project(12)),
            ('rproject(24)', random_project(24)),
            ('rproject(128, ridge(.01))', random_project(128, ridge_regression(.01))),
            ('rproject(128, lasso(.01))', random_project(128, lasso_regression(.01))),
            ]:
        results['{0}-{1}'.format(lname,name)] = leave1out(learner, features, labels)
        results['{0}-normed-{1}'.format(lname,name)] = leave1out(learner, features, normlabels(labels))
        learner = norm_learner(learner)
        results['{0}-normed1-{1}'.format(lname,name)] = leave1out(learner, features, labels)

print_results(results)
