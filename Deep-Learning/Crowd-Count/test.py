import skimage.io
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import skimage.io
from skimage.color import rgb2gray
import scipy.ndimage
import torchvision
from src.models import inception
import torch
from torchsummary import summary
import cv2


def gaussian_filter_density(gt, pts):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = len(pts)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        try:
            pt2d[int(math.floor(pt[1])), int(math.floor(pt[0]))] = 1.
            print(gt.shape, math.floor(pt[1]), math.floor(pt[0]))
        except:
            pass
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


a = np.load("./data/preprocessed/test_density/IMG_1.npy")
plt.imshow(a)
plt.show()
