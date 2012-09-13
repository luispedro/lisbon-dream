import numpy as np
class random_result(object):
    def __init__(self, dim, R):
        self.dim = dim
        self.R = R

    def apply_one(self, _):
        return self.R.rand(self.dim)

class random_learner(object):
    '''
    This does not actually learn anything. It just returns random values
    '''
    def __init__(self, seed):
        from milk.utils.utils import get_nprandom
        self.R = get_nprandom(seed)

    def train(self, _, labels):
        return random_result(len(labels[0]), self.R)
