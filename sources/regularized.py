import numpy as np

from projection import product_intercept_predictor

class ridge_regression(object):
    '''
    Perform a random projection to ``nr_dims`` dimensions and then fit a
    least-squares model on this space

    '''
    def __init__(self, alpha=.128):
        self.alpha = alpha

    def train(self, features, labels):
        from sklearn import linear_model
        labels = labels.copy()
        labels[np.isnan(labels)] = 0
        clf = linear_model.Ridge(alpha=self.alpha)
        clf.fit(features, labels)
        return product_intercept_predictor(clf.coef_.T.copy(), clf.intercept_)


class lasso_regression(object):
    '''
    Perform a random projection to ``nr_dims`` dimensions and then fit a
    least-squares model on this space

    '''
    def __init__(self, alpha=.128):
        self.alpha = alpha

    def train(self, features, labels):
        from sklearn import linear_model
        labels = labels.copy()
        labels[np.isnan(labels)] = 0
        clf = linear_model.Lasso(alpha=self.alpha)
        clf.fit(features, labels)
        return product_intercept_predictor(clf.coef_.T.copy(), clf.intercept_)


