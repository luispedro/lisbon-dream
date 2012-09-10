from jug import Task, TaskGenerator
from leave1out import leave1out
from projection import random_project
from randomlearner import random_learner
from regularized import ridge_regression, lasso_regression
from preproc import preproc, rna_seq_active_only, zscore1
import numpy as np

leave1out = TaskGenerator(leave1out)
@TaskGenerator
def print_results(results):
    with open('results.txt', 'w') as output:
        for name in sorted(results.keys()):
            print >>output, '{0:<40}: {1}'.format(name, int(100*results[name]))


def zscore_rna_seq():
    features, labels = rna_seq_active_only()
    return zscore1(features), labels

results = {}
for lname,loader in [
                ('rna+ge', preproc),
                ('active', rna_seq_active_only),
                ('active+zscore', zscore_rna_seq),
                ]:
    preprocessed = Task(loader)
    features = preprocessed[0]
    labels = preprocessed[1]

    for name,learner in [
            ('ridge', ridge_regression()),
            ('lasso(.05)', lasso_regression(.05)),
            ('lasso(.5)', lasso_regression(.5)),
            ('lasso(.8)', lasso_regression(.8)),
            ('lasso(2)', lasso_regression(2)),
            ('random', random_learner()),
            ('rproject(12)', random_project(12)),
            ]:
        results['{0}-{1}'.format(lname,name)] = leave1out(learner, features, labels)

print_results(results)
