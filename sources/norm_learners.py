from milk.supervised.base import supervised_model
import numpy as np
class add_model(supervised_model):
    def __init__(self, base, b):
        self.base = base
        self.b = b

    def apply(self, features):
        return self.base.apply(features) + self.b

class norm_learner(object):
    def __init__(self, base, axis):
        self.base = base
        self.axis = axis

    def train(self, features, labels):
        from milk.unsupervised import center
        labels, mean = center(labels, axis=self.axis)
        return self.base.train(features, labels)

class znorm_learner(object):
    def __init__(self, base, axis):
        self.base = base
        self.axis = axis

    def train(self, features, labels):
        from milk.unsupervised import zscore
        labels = zscore(labels, axis=self.axis)
        return self.base.train(features, labels)

class rank_learner(object):
    def __init__(self, base, axis=0):
        self.base = base
        self.axis = axis

    def __repr__(self):
        return 'rank_learner({0})'.format(self.base)

    __str__ = __repr__

    def train(self, features, labels):
        from milk.unsupervised import zscore
        from scipy import stats
        if self.axis == 0:
            rlabels = np.array([stats.rankdata(ells) for ells in labels])
        else:
            rlabels = np.array([stats.rankdata(ells) for ells in labels.T])
            rlabels = rlabels.T
        rlabels[np.isnan(labels)] = np.nan
        return self.base.train(features, rlabels)

