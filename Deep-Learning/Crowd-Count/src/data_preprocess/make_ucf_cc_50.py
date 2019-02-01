# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2019-02
# ------------------------

import h5py
import scipy.io as io
import glob
import warnings
import os
import numpy as np
import skimage.io
from gaussian_filter import gaussian_filter_density

warnings.filterwarnings("ignore")

path = "../../data/ucf_cc_50"

for i, img_path in enumerate(glob.glob(os.path.join(path, '*.jpg'))):
    print(str(i) + " / 50")
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