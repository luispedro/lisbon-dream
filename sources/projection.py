import numpy as np

class product_predictor(object):
    def __init__(self, x):
        self.x = x

    def apply(self, v):
        return np.dot(v, self.x)

class product_intercept_predictor(object):
    def __init__(self, x, beta):
        self.x = x
        self.beta = beta

    def apply(self, v):
        return np.dot(v, self.x) + self.beta

class least_squares(object):
    def train(self, features, labels):
        nr_celltypes,nr_features = features.shape
        nr_celltypes_prime,nr_drugs = labels.shape
        assert nr_celltypes_prime == nr_celltypes
        xs = []
        for ci in xrange(nr_drugs):
            clabels = labels[:,ci]
            active = ~np.isnan(clabels)
            x,residues,rank,s = np.linalg.lstsq(features[active], labels[active,ci])
            xs.append(x)
        return product_predictor(np.array(xs).T)

class random_project(object):
    '''
    Perform a random projection to ``nr_dims`` dimensions and then fit a
    least-squares model on this space

    '''
    def __init__(self, nr_dims=12, learner=None):
        self.nr_dims = nr_dims
        if learner is None:
            self.learner = least_squares()
        else:
            self.learner = learner

    def train(self, features, labels):
        from milk.supervised.classifier import ctransforms_model
        nr_celltypes,nr_features = features.shape
        nr_celltypes_prime,nr_drugs = labels.shape
        assert nr_celltypes_prime == nr_celltypes
        V = np.random.rand(nr_features, self.nr_dims)
        features = np.dot(features, V)
        features /= features.mean()
        inner = self.learner.train(features, labels)
        return ctransforms_model([product_predictor(V), inner])
