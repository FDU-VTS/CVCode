# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

PART_A = "./data/part_A_final"
PART_B = "./data/part_B_final"


class ShanghaiTechDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


train_ground = os.path.join(PART_A, "train_data/ground_truth")
ground_image = os.path.join(PART_A, "train_data/images")
for i in range(1, 2):
    mat_path = os.path.join(train_ground, "GT_IMG_{0}".format(i))
    image_path = os.path.join(ground_image, "IMG_{0}.jpg".format(i))
    mat = loadmat(mat_path)
    image = skimage.io.imread(image_path)
    print(len(mat["image_info"][0][0][0][0][0]))
    plt.show()