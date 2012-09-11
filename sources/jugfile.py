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
    features, labels = rna_seq_active_only()
    return zscore1(features), labels

def rna_ge_gosweigths_pruned():
    features, labels = rna_ge_gosweigths()
    return prune_similar(features, frac=.5), labels

results = {}
for lname,loader in [
                ('rna+ge', preproc),
                ('active', rna_seq_active_only),
                ('active+zscore', zscore_rna_seq),
                ('rna+ge+active+zscore', ge_rna_valid),
                ('gosweigths', rna_ge_gosweigths),
                ('prune(gosweigths, .5)', rna_ge_gosweigths_pruned),
                ]:
    preprocessed = Task(loader)
    features = preprocessed[0]
    labels = preprocessed[1]

    for name,learner in [
            ('ridge', ridge_regression()),
            ('ridge(.001)', ridge_regression(.001)),
            ('ridge(1.)', ridge_regression(1.)),
            ('lasso(.000000002)', lasso_regression(.000000002)),
            ('lasso(.0002)', lasso_regression(.0002)),
            ('lasso(.05)', lasso_regression(.05)),
            ('lasso(.5)', lasso_regression(.5)),
            ('lasso(.8)', lasso_regression(.8)),
            ('lasso(2)', lasso_regression(2)),
            ('random', random_learner()),
            ('rproject(12)', random_project(12)),
            ('rproject(24)', random_project(24)),
            ('rproject(128, ridge(.01))', random_project(128, ridge_regression(.01))),
            ('rproject(128, lasso(.01))', random_project(128, lasso_regression(.01))),
            ]:
        results['{0}-{1}'.format(lname,name)] = leave1out(learner, features, labels)

print_results(results)
