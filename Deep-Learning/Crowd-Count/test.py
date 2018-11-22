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


# path = "./data/original/part_A_final/train_data/ground_truth/GT_IMG_141.mat"
# image_name = "./data/original/part_A_final/train_data/images/IMG_141.jpg"
# image = skimage.io.imread(image_name)
# if image.shape[-1] == 3:
#     image = rgb2gray(image)
# mat = loadmat(path)
# image_info = mat["image_info"]
# ann_points = image_info[0][0][0][0][0]
# ann_points = sorted(ann_points, key=lambda x:x[0])
# for i in ann_points:
#     print(i)
# gaussian_filter_density(image, ann_points)

