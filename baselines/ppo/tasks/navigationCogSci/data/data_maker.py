import numpy as np
from PIL import Image
import glob
import pickle

def grayscale(pix):
    pix = 0.299 * pix[:, 0] + 0.587 * pix[:, 1] + 0.114 * pix[:, 2]
    pix = pix.astype(np.float32) / 255
    return pix.astype(np.float32)

res = dict()
NUM_DEGS = 15

for image_path in glob.glob("*.png"):
    image = np.array(Image.open(image_path).getdata())
    image = np.reshape(grayscale(image), [84, 84, 1])
    _, obj, deg = image_path[:-4].split('_')
    deg = NUM_DEGS * int(deg[3:]) // 360
    if obj not in res:
        res[obj] = np.zeros([NUM_DEGS, 84, 84, 1], np.float32)
    res[obj][deg] = image
    print(image.shape, image.dtype, image.min(), image.max())

for obj in res.keys():
    print(res[obj].shape, res[obj].dtype, obj)

with open('data.pkl', 'wb') as F:
    pickle.dump(res, F)
