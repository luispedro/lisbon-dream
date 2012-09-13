from milk.supervised.base import supervised_model
import numpy as np
def select1(fi, ells):
    C = np.array([np.cov(fi[:,i],ells)[0,1] for i in xrange(len(fi.T))])
    R = np.array([np.cov(np.random.rand(len(ells)),ells)[0,1] for i in xrange(10000)])
    R = np.abs(R)
    R.sort()
    cutoff = R[-len(R)//100]
    return np.abs(C) > cutoff


def select(features, labels):
    ss = []
    for ells in labels.T:
        active = ~np.isnan(ells)
        ss.append(select1(features[active,:], ells[active]))
    ss = np.array(ss)
    return (ss.sum(0) > 20)

class select_model(supervised_model):
    def __init__(self, mask):
        self.mask = mask

    def apply(self, features):
        return features[self.mask]

class select_learner(object):
    def train(self, features, labels):
        return select_model(select(features, labels))

