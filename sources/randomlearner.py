import numpy as np
class random_result(object):
    def __init__(self, dim):
        self.dim = dim

    def apply_one(self, _):
        return np.random.rand(self.dim)

class random_learner(object):
    def train(self, _, labels):
        return random_result(len(labels[0]))
