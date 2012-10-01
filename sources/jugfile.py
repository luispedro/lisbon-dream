from jug import Task, TaskGenerator
from leave1out import leave1out
from projection import random_project, random2_project
from projection import *
from randomlearner import random_learner
from selectlearner import *
from regularized import ridge_regression, lasso_regression, lasso_regression_with_learning
from regularized import lasso_relaxed_after_learning
from regularized import lasso_relaxed_after_learning, lasso_relaxed
from regularized import *
from preproc import *
from norm_learners import *
import numpy as np
from milk.supervised.classifier import ctransforms
from print_by_data import print_detailed_results
from milk.supervised.normalise import zscore_normalise

leave1out = TaskGenerator(leave1out)
@TaskGenerator
def print_results(results):
    with open('results/results.txt', 'w') as output:
        for name in sorted(results.keys(), key=lambda k: np.mean(results[k][0])):
            val = results[name]
            val = np.mean(val[0])
            print >>output, '{0:<64}: {1: .2%}'.format('{0}-{1}-{2}'.format(*name), val)




@TaskGenerator
def prune_features(data, frac):
    features = data[0]
    labels = data[1]
    return prune_similar(features, frac=frac), labels

@TaskGenerator
def thresh(data):
    features = data[0]
    labels = data[1]
    return thresh_features(features), labels


@TaskGenerator
def concat_features(data0,data1):
    features0 = data0[0]
    features1 = data1[0]
    labels = data0[1]
    return np.hstack((features0,features1)), labels


rna_ge_gosweigths_add = rna_ge_gosweigths('add')
rna_ge_gosweigths_mp_add = rna_ge_gosweigths('add', ['molecular_function'])
rna_ge_gosweigths_mp_ma = rna_ge_gosweigths('maxabs', ['molecular_function'])
rna_ge_gosweigths_mpbf_add = rna_ge_gosweigths('add', ['molecular_function', 'biological_process'])
rna_ge_gosweigths_mpbf_ma = rna_ge_gosweigths('maxabs', ['molecular_function', 'biological_process'])
rna_ge_gosweigths_bf_add = rna_ge_gosweigths('add', ['biological_process'])
rna_ge_gosweigths_bf_ma = rna_ge_gosweigths('maxabs', ['biological_process'])
rna_ge_gosweigths_maxabs = rna_ge_gosweigths('maxabs')
ge_rna_valid_mean = Task(ge_rna_valid)

results = {}
for lname,data in [
                ('rna+ge+act+zs', ge_rna_valid_mean),
                ('thresh(rna+ge+act+zs)', thresh(ge_rna_valid_mean)),
                ('prune(thresh(rna+ge+act+zs))', prune_features(thresh(ge_rna_valid_mean),.5)),
                ('prune(rna+ge+act+zs)', prune_features(ge_rna_valid_mean,.5)),
                ('rna+ge+act(ma)+zs', Task(ge_rna_valid,'maxabs')),
                ('gow', rna_ge_gosweigths_add),
                ('gow-thresh', thresh(rna_ge_gosweigths_add)),
                ('gowmp', rna_ge_gosweigths_mp_add),
                ('gowmp-ma', rna_ge_gosweigths_mp_ma),
                ('thresh(gowmp)', thresh(rna_ge_gosweigths_mp_add)),
                ('thresh(gowmp-ma)', thresh(rna_ge_gosweigths_mp_ma)),
                ('gowmpbf', rna_ge_gosweigths_mpbf_add),
                ('gowmpbf-ma', rna_ge_gosweigths_mpbf_ma),
                ('thresh(gowmpbf)', thresh(rna_ge_gosweigths_mpbf_add)),
                ('thresh(gowmpbf-ma)', thresh(rna_ge_gosweigths_mpbf_ma)),
                ('gowbf', rna_ge_gosweigths_bf_add),
                ('thresh(gowbf)', thresh(rna_ge_gosweigths_bf_add)),
                ('gow-ma', rna_ge_gosweigths_maxabs),
                ('gow-ma-thresh', thresh(rna_ge_gosweigths_maxabs)),
                ('prune(gow, .5)', prune_features(rna_ge_gosweigths_add, .5)),
                ('prune(gowmp, .5)', prune_features(rna_ge_gosweigths_mp_add, .5)),
                ('thresh(gowmp)', thresh(rna_ge_gosweigths_mp_add)),
                ('CONCAT thresh(gowmp-ma)+prune(rna+ge+act)', concat_features(thresh(rna_ge_gosweigths_mp_ma), prune_features(ge_rna_valid_mean,.5))),
                ('CONCAT thresh(gowmp)+prune(rna+ge+act)', concat_features(thresh(rna_ge_gosweigths_mp_add), prune_features(ge_rna_valid_mean,.5))),
                ('prune(gowmpbf, .5)', prune_features(rna_ge_gosweigths_mpbf_add, .5)),
                ('prune(thresh(gowmpbf), .5)', prune_features(thresh(rna_ge_gosweigths_mpbf_add), .5)),
                ('prune(thresh(gowmpbf), .1)', prune_features(thresh(rna_ge_gosweigths_mpbf_add), .1)),
                ('prune(thresh(gowbf), .5)', prune_features(thresh(rna_ge_gosweigths_bf_add), .5)),
                ('prune(thresh(gowbf), .1)', prune_features(thresh(rna_ge_gosweigths_bf_add), .1)),
                ('prune(gow-ma, .5)', prune_features(rna_ge_gosweigths_maxabs, .5)),
                ('prune(gow, .9)', prune_features(rna_ge_gosweigths_add, .9)),
                ('prune(gow-ma, .9)', prune_features(rna_ge_gosweigths_maxabs, .9)),
                ('prune(gow, .95)', prune_features(rna_ge_gosweigths_add, .95)),
                ('prune(gow-ma, .95)', prune_features(rna_ge_gosweigths_maxabs, .95)),
                ]:
    features = data[0]
    labels = data[1]

    for name,learner in [
            ('random(1)', random_learner(1)),
            ('random(2)', random_learner(2)),
            ('random(3)', random_learner(3)),

            ('ridge(1.)', ridge_regression(1.)),
            ('ridge(.128)', ridge_regression(.128)),
            ('ridge(.001)', ridge_regression(.001)),
            ('ridge(1e-6)', ridge_regression(1e-6)),
            ('select+ridge(.001)', ctransforms(remove_constant_features(), select_learner(12), ridge_regression(.001))),

            ('relaxed', lasso_relaxed(.000225010113525, .1)),
            ('sel-relaxed', ctransforms(remove_constant_features(), select_learner(12), lasso_relaxed(.000225010113525, .1))),
            ('zs-sel-relaxed', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_relaxed(.000225010113525, .1))),
            ('sel+lasso(1e-7)', ctransforms(remove_constant_features(), select_learner(12), lasso_regression(1e-7))),
            ('sel+lasso(1e-5)', ctransforms(remove_constant_features(), select_learner(16), lasso_regression(1e-5))),
            ('zs+lasso(1e-5)', ctransforms(remove_constant_features(), zscore_normalise(), lasso_regression_guess())),
            ('min0+lasso(1e-5)', ctransforms(remove_constant_features(), min_to_zero(), lasso_regression_guess())),
            ('sel+lasso(0.000225010113525)', ctransforms(remove_constant_features(), select_learner(16), lasso_regression(0.000225010113525))),
            ('zs+sel+lasso(0.000225010113525)', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(16), lasso_regression(0.000225010113525))),
            ('zs+sel+lasso_learn', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_regression_with_learning())),
            ('zs+sel+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_regression_guess())),
            ('zs+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), lasso_regression_guess())),
            ('zs+sel+min0+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), min_to_zero(), lasso_regression_guess())),
            ('zs+min0+lasso_guess', ctransforms(remove_constant_features(), zscore_normalise(), min_to_zero(), lasso_regression_guess())),
            ('min0+lasso_guess', ctransforms(remove_constant_features(), min_to_zero(), lasso_regression_guess())),
            ('lasso_guess', ctransforms(remove_constant_features(), lasso_regression_guess())),
            ('min0+lasso_select+ridge', ctransforms(remove_constant_features(), min_to_zero(), lasso_selection(0.000225010113525), ridge_regression(.01))),

            ('rproject(8)', random_project(8)),
            ('rproject(12)', random_project(12)),
            ('rproject(14)', random_project(14)),
            ('rproject(16)', random_project(16)),
            ('rproject(18)', random_project(18)),
            ('rproject(24)', random_project(24)),

            ('r2project(8)', random2_project(8)),
            ('r2project(12)', random2_project(12)),
            ('r2project(14)', random2_project(14)),
            ('r2project(16)', random2_project(16)),
            ('r2project(18)', random2_project(18)),
            ('r2project(24)', random2_project(24)),

            ('rpproject(8)', randomp_project(8)),
            ('rpproject(12)', randomp_project(12)),
            ('rpproject(14)', randomp_project(14)),
            ('rpproject(16)', randomp_project(16)),
            ('rpproject(18)', randomp_project(18)),
            ('rpproject(24)', randomp_project(24)),

            ('rp2project(12)', randomp2_project(12)),
            ('rp2project(24)', randomp2_project(24)),

            ('rs2project(12)', randoms2_project(12)),
            ('rs2project(24)', randoms2_project(24)),
            ('rs3project(24)', randoms3_project(24)),

            ('rs2project(64, ridge(.01))', randoms2_project(64, ridge_regression(.01))),
            ('rs2project(64, lasso_guess)', randoms2_project(64, lasso_regression_guess())),
            ('rs2project(128, lasso_guess)', randoms2_project(128, lasso_regression_guess())),



            ('rproject(16, ridge(.01))', random_project(16, ridge_regression(.01))),
            ('rproject(64, ridge(.01))', random_project(64, ridge_regression(.01))),
            ('rpproject(64, ridge(.01))', randomp_project(64, ridge_regression(.01))),
            ('rproject(128, ridge(.01))', randomp_project(128, ridge_regression(.01))),
            ('rp2roject(128, ridge(.01))', randomp2_project(128, ridge_regression(.01))),
            ('rproject(256, ridge(.01))', random_project(256, ridge_regression(.01))),
            ('rproject(1024, lasso_learn)', random_project(1024, lasso_regression_with_learning())),
            ('rproject(128, lasso_guess)', random_project(128, lasso_regression_guess())),
            ('rproject(256, lasso_guess)', random_project(256, lasso_regression_guess())),
            ('rproject(1024, lasso_guess)', random_project(1024, lasso_regression_guess())),
            ]:
        results[lname,'raw',name] = leave1out(learner, features, labels)
        learner0 = norm_learner(learner, 0)
        learner1 = norm_learner(learner, 1)
        results[lname,'normed0',name] = leave1out(learner0, features, labels)
        results[lname,'normed1',name] = leave1out(learner1, features, labels)
        zlearner0 = znorm_learner(learner, 0)
        zlearner1 = znorm_learner(learner, 1)
        results[lname,'znormed0',name] = leave1out(zlearner0, features, labels)
        results[lname,'znormed1',name] = leave1out(zlearner1, features, labels)
        rlearner0 = rank_learner(learner,0)
        rlearner1 = rank_learner(learner,1)
        results[lname,'ranked0',name] = leave1out(rlearner0, features, labels)
        results[lname,'ranked1',name] = leave1out(rlearner1, features, labels)

print_results(results)
print_detailed_results(results)
