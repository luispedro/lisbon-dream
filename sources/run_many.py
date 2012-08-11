from leave1out import leave1out
from projection import random_project
from randomlearner import random_learner
from preproc import preproc
import numpy as np
np.random.seed(322)

features,labels = preproc()


for name,learner in [
        ('random', random_learner()),
        ('random_project(12)', random_project(12)),
        ]:
    val = leave1out(learner, features, labels)
    print '%-24s: %s' % (name, val)