from scipy import stats
import numpy as np

def spearnan_compare(predicted, gold):
    assert len(predicted) == len(gold)
    active = ~np.isnan(gold)
    if predicted[active].ptp() == 0 or gold[active].ptp() == 0:
        return (0.,1.)
    c,p = stats.spearmanr(predicted[active], gold[active])
    assert not np.isnan(c)
    assert not np.isnan(p)
    return c,p

def leave1out(learner, features, labels):
    '''
    corrs_ps = leave1out(learner, features, labels)

    Perform leave-1-out cross-validation on cell types
    '''
    predicted = []
    for i in xrange(len(labels)):
        idx = np.ones(len(labels), bool)
        idx[i] = 0
        model = learner.train(features[idx], labels[idx])
        predicted.append(model.apply(features[i]))
    predicted = np.array(predicted)

    corrs = []
    pvals = []
    for p,ell in zip(predicted.T, labels.T):
        corr,ps = spearnan_compare(p, ell)
        corrs.append(corr)
        pvals.append(ps)
    return np.array([corrs,ps])
