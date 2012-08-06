import numpy as np
class product_predictor(object):
    def __init__(self, x):
        self.x = x

    def apply_one(self, v):
        return np.dot(v, self.x)

class random_project(object):
    def __init__(self, nr_dims=12):
        self.nr_dims = nr_dims
    def train(self, features, labels):
        n,m = features.shape
        V = np.random.rand(m, self.nr_dims)
        features = np.dot(features, V)
        features /= features.mean()
        xs = []
        for i in xrange(n):
            x,residues,rank,s = np.linalg.lstsq(features, labels[:,i])
            xs.append(x)
        xs = np.array(xs)
        return product_predictor(np.dot(V,xs.T))
