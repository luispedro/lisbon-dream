from milk.supervised import lasso_learner
from milk.supervised import lasso_model_walk

from projection import product_intercept_predictor
import numpy as np

class lasso_regression(object):
    def __init__(self, lam):
        self.learner = lasso_learner(lam)

    def train(self, features, labels):
        return self.learner.train(features.T, labels.T)

def _learn(base, features, labels, alpha):
    nr_celltypes,nr_features = features.shape
    nr_celltypes_prime,nr_drugs = labels.shape
    assert nr_celltypes_prime == nr_celltypes
    betas = []
    xs = []
    for ci in xrange(nr_drugs):
        clabels = labels[:,ci]
        active = ~np.isnan(clabels)
        clf = base(alpha)
        clf.fit(features[active], labels[active,ci])
        xs.append(clf.coef_.T.copy())
        betas.append(clf.intercept_.copy())
    return product_intercept_predictor(np.array(xs).T, np.array(betas))


class ridge_regression(object):
    '''
    Perform a random projection to ``nr_dims`` dimensions and then fit a
    least-squares model on this space

    '''
    def __init__(self, alpha=.128):
        self.alpha = alpha

    def train(self, features, labels):
        from sklearn import linear_model
        return _learn(linear_model.Ridge, features, labels, self.alpha)


class lasso_path_regression(object):

    def train(self, features, labels):
        from sklearn import linear_model
        betas = []
        xs = []
        for ci,ells in enumerate(labels.T):
            active = ~np.isnan(ells)
            fi = features[active]
            ells = ells[active]
            fits = linear_model.lasso_path(fi, ells)
            xs.append(fits[-1].coef_.T.copy())
            betas.append(fits[-1].intercept_.copy())
        return product_intercept_predictor(np.array(xs).T, np.array(betas))


def select_lam(features, labels):
    from leave1out import spearnan_compare
    predicted = []
    for i in xrange(len(labels)):
        idx = np.ones(len(labels), bool)
        idx[i] = 0
        models,lams = lasso_model_walk(features[idx].T, labels[idx].T, start=.1, step=.72, nr_steps=80, return_lams=True)
        predicted.append([model.apply(features[i].T) for model in models])
    predicted = np.array(predicted)
    best = None
    bestval = -8.
    allvalues = []
    for li,lam in enumerate(lams):
        cur = 0.0
        for p,ell in zip(predicted[:,li,:].T, labels.T):
            corr,ps = spearnan_compare(p, ell)
            cur += corr
        cur /= labels.shape[1]
        allvalues.append(cur)
        if cur > bestval:
            best = lam
            bestli = li
            bestval = cur
    return best, bestval

class lasso_regression_with_learning(object):
    def train(self, features, labels):
        best, val = select_lam(features, labels)
        learner = lasso_learner(best)
        return learner.train(features.T, labels.T)

