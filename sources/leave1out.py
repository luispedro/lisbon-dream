from scipy import stats
from random_predictor import *

def leave1out(learner, features, labels):
    '''
    avg_corr = leave1out(learner, features, labels):
    '''
    corrs = []
    for i in xrange(len(labels)):
        idx = np.ones(len(labels), bool)
        idx[i] = 0
        p = learner.train(features[idx], labels[idx])
        corr,ps = stats.spearmanr(p.apply_one(features[i]), labels[i])
        corrs.append(corr)
    return np.mean(corrs)
