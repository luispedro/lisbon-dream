import numpy as np
from leave1out import leave1out
class walk(object):
    def __init__(self, base):
        self.base = base

    def train(self, features, labels):
        def f(a):
            v = np.mean(leave1out(self.base(alpha), features, labels)[0])
            print 'f({0}) => {1}'.format(a,v)
            return v
        direction = 0
        step = 0.1
        alpha = 1.
        while direction == 0:
            val = f(alpha)
            val1 = f(alpha + alpha*step)
            delta = val1 - val
            if delta > 0:
                direction = +1
                val = val1
            elif delta < 0:
                direction == -1
                val = f(alpha - alpha*step)
            elif alpha > 0.00000000001:
                alpha /= 2.
            else:
                learner = self.base(1.)
                return learner.train(features, labels)
        while True:
            step *= 2
            n = alpha + direction*step*direction
            val1 = f(n)
            if val > val1:
                step /= 2.
                n = alpha + direction*step*direction
                learner = self.base(alpha)
                return learner.train(features, labels)
                

