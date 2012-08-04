import numpy as np
from projection import *
def test_smoke():
    learner = random_project()
    features = np.random.rand(24,1200)
    labels = np.random.rand(24,30)
    model = learner.train(features, labels)
    val = model.apply_one(np.random.rand(1200))
    assert val.shape == (30,)
