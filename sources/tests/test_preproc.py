from preproc import maxabs
import numpy as np

def test_maxabs():
    np.random.seed(232)
    A =  np.random.random_sample((2,12))
    assert np.all(maxabs(A) == A.max(0))
