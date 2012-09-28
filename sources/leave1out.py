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

def leave1out(learner, features, labels, step=3):
    '''
    corrs_ps = leave1out(learner, features, labels)

    Perform leave-1-out cross-validation on cell types
    '''
    features = np.asanyarray(features)
    predicted = []
    for i in xrange(0,len(labels),3):
        idx = np.ones(len(labels), bool)
        idx[i:i+3] = 0
        model = learner.train(features[idx], labels[idx])
        for j in xrange(i, min(len(features), i + 3)):
            predicted.append(model.apply(features[j]))
    predicted = np.array(predicted)

    corrs = []
    pvals = []
    for p,ell in zip(predicted.T, labels.T):
        corr,ps = spearnan_compare(p, ell)
        corrs.append(corr)
        pvals.append(ps)
    return np.array([corrs,ps])
