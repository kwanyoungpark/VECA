import numpy as np
import pickle

with open('data.pkl', 'rb') as F:
    data = pickle.load(F)

for obj in data.keys():
    print(data[obj].shape, data[obj].dtype, obj)
