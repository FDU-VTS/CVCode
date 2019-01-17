# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2019-01
# ------------------------

import h5py
import scipy.io as io
import glob
from scipy.ndimage.filters import gaussian_filter
import scipy
import math
import warnings
import os
import numpy as np
import skimage.io
warnings.filterwarnings("ignore")


#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    num = pts.shape[0] - 1
    for i, pt in enumerate(pts):
        print(str(i) + " / " + str(num))
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[math.floor(pt[1]), math.floor(pt[0])] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


root = '../../data/UCF-QNRF_ECCV18/'
train_path = os.path.join(root, 'Train')
test_path = os.path.join(root, 'Test')
path_sets = [train_path, test_path]

for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        mat = io.loadmat(img_path.replace('.jpg','_ann.mat'))
        img = skimage.io.imread(img_path, plugin='matplotlib')
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["annPoints"]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg','.h5'), 'w') as hf:
            hf['density'] = k