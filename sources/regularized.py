import numpy as np

from projection import product_intercept_predictor

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

class lasso_regression(object):
    '''

    '''
    def __init__(self, alpha=.128):
        self.alpha = alpha

    def train(self, features, labels):
        from sklearn import linear_model
        return _learn(linear_model.Lasso, features, labels, self.alpha)

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

class lars_regression(object):
    '''

    '''
    def __init__(self, nr_coeffs):
        self.nr_coeffs = nr_coeffs

    def train(self, features, labels):
        from sklearn import linear_model
        return _learn(linear_model.Lars, features, labels, self.nr_coeffs)

