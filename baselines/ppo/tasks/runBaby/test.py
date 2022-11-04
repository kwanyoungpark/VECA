from utils import momentsTracker as Tracker
import numpy as np

tracker = Tracker(2, 0.)
data = np.concatenate([np.random.rand(128, 1), np.ones((128, 1))], axis = 1)
for i in range(128):
    tracker.update(data[i])
    print(tracker.moments())

mean, var = tracker.moments()
data = (data - mean) / (var + 1e-8)
print(np.mean(data, axis = 0), np.var(data, axis = 0))
