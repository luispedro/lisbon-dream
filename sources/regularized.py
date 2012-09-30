from milk.supervised import lasso
from milk.supervised.lasso import lasso_model
from milk.supervised import lasso_learner
from milk.supervised import lasso_model_walk

from projection import product_intercept_predictor
import numpy as np

class lasso_regression(object):
    def __init__(self, lam):
        self.learner = lasso_learner(lam)

    def train(self, features, labels):
        return self.learner.train(features.T, labels.T)

class lasso_regression_guess(object):

    def train(self, features, labels):
        labelsf = np.nan_to_num(labels)
        val = np.dot(features.T, labelsf)
        lam = 1e-3*np.abs(val).max()/float(labels.size)
        print('Guessed {0}.'.format(lam))
        learner = lasso_learner(lam)
        model = learner.train(features.T, labels.T)
        return model

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
    from milk.unsupervised import center
    from scipy import optimize
    def best_B(i, lam):
        best = None
        bestval = np.inf
        for k in models[i]:
            if np.abs(k-lam) < np.abs(k-bestval):
                bestval = k
                best = models[i][k]
        if best is not None:
            return best.copy(), bestval
        return None, bestval
    def evaluate(lam):
        predicted = []
        for i in xrange(0,len(labels),3):
            idx = np.ones(len(labels), bool)
            idx[i:i+3] = 0
            Y, Ymean = center(labels[idx].T, axis=1)
            B,err = best_B(i,lam)
            if err > 0:
                B = lasso(features[idx].T, Y, B, lam=lam, tol=1e-3)
            model = lasso_model(B, Ymean)
            predicted.append(model.apply(features[i].T))
            try:
                predicted.append(model.apply(features[i+1]))
                predicted.append(model.apply(features[i+2]))
            except IndexError:
                pass
            models[i][lam] = B
        predicted = np.array(predicted)
        cur = 0.0
        for p,ell in zip(predicted.T, labels.T):
            corr,ps = spearnan_compare(p, ell)
            cur += corr
        print('evaluate({0}) = {1:.2%}'.format(lam, cur / labels.shape[1]))
        return - cur / labels.shape[1]

    features = np.asanyarray(features)
    labels = np.asanyarray(labels)
    models = [{} for _ in labels]
    start = 1.
    last = 128.
    seen = []
    lams = []
    for it in xrange(24):
        cur = evaluate(start)
        lams.append(start)
        seen.append(cur)
        if cur > last:
            break
        start /= 4.
        last = cur
    else:
        return last

    if seen[-2] < .05:
        brack = (lams[-1], lams[-2], lams[-3])
        if seen[-2] >= seen[-3]:
            brack = (lams[-1], lams[-2])
        val = optimize.brent(evaluate, brack=brack, maxiter=16)
        print('Best found ({0}) has value {1:.2%}'.format(val, -evaluate(val)))
        return val
    else:
        print('Skipping detailed optimisation ({0} has value {1})'.format(lams[-2], -seen[-2]))
        return lams[-2]

class lasso_regression_with_learning(object):
    def train(self, features, labels):
        best = select_lam(features, labels)
        learner = lasso_learner(best)
        return learner.train(features.T, labels.T)
class lasso_relaxed_after_learning(object):
    def train(self, features, labels):
        from milk.supervised.classifier import ctransforms_model
        from selectlearner import select_model
        best = select_lam(features, labels)
        learner = lasso_learner(best)
        model = learner.train(features.T, labels.T)
        betas = model.betas
        active = np.abs(betas) > 1.e-8
        print active.sum()
        active = active.any(0)
        print active.sum()
        print features.shape
        features = features[:,active]
        print features.shape

        best = select_lam(features, labels)
        learner = lasso_learner(best)
        model = learner.train(features.T, labels.T)
        return ctransforms_model([select_model(active),model])

class lasso_relaxed(object):
    def __init__(self, lam1, phi):
        self.lam1 = lam1
        self.phi = phi
    def train(self, features, labels):
        from milk.supervised.classifier import ctransforms_model
        from selectlearner import select_model
        learner = lasso_learner(self.lam1)
        model = learner.train(features.T, labels.T)
        betas = model.betas
        active = np.abs(betas) > 1.e-8
        print active.sum()
        active = active.any(0)
        print active.sum()
        print features.shape
        features = features[:,active]
        print features.shape

        learner = lasso_learner(self.lam1 * self.phi)
        model = learner.train(features.T, labels.T)
        return ctransforms_model([select_model(active),model])

