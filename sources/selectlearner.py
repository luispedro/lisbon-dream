from milk.supervised.base import supervised_model
import numpy as np

def corrcoefs(X, y):
    '''
    Cs = corrcoefs(X, y)

    Cs[i] = np.corrcoef(X[i], y)[0,1]
    '''
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    xy = np.dot(X, y)
    y_ = y.mean()
    ys_ = y.std()
    x_ = X.mean(1)
    xs_ = X.std(1)
    n = float(len(y))
    ys_ += 1e-5 # Handle zeros in ys
    xs_ += 1e-5 # Handle zeros in x

    return (xy - x_*y_*n)/n/xs_/ys_



def select1(fi, ells, R):
    C = corrcoefs(fi.T, ells)
    R = corrcoefs(R.rand(10000, len(ells)), ells)
    R = np.abs(R)
    R.sort()
    cutoff = R[-len(R)//100]
    return np.abs(C) > cutoff


def select(features, labels, R):
    from milk.utils.utils import get_nprandom
    R = get_nprandom(R)
    ss = []
    for ells in labels.T:
        active = ~np.isnan(ells)
        ss.append(select1(features[active,:], ells[active], R))
    ss = np.array(ss)
    for cutoff in xrange(20,0):
        selected = (ss.sum(0) > cutoff)
        if np.sum(selected) > 10:
            return selected
    return np.ones(features.shape[1], bool)

class select_model(supervised_model):
    def __init__(self, mask):
        self.mask = mask

    def apply(self, features):
        return features[self.mask]

class select_learner(object):
    def __init__(self, R):
        self.R = R
    def train(self, features, labels):
        return select_model(select(features, labels, self.R))

class remove_constant_features(object):
    def train(self, features, labels):
        from milk.supervised.featureselection import filterfeatures
        constant = features.ptp(0) != 0
        return filterfeatures(constant)


class subtract_model(supervised_model):
    def __init__(self, val):
        self.val = val
    def apply(self, features):
        return features - self.val
class min_to_zero(object):
    def train(self, features, labels):
        features = features.copy()
        return subtract_model(features.min(0))

