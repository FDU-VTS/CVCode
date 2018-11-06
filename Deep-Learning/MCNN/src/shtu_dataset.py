# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from torch.utils.data import Dataset
import os
import skimage.io
import skimage.transform
from skimage.color import rgb2gray
import numpy as np
import pandas as pd


def get_data(mode="train"):
    # index train_image ground_truth
    data_path = "./data/preprocessed/{0}".format(mode) \
        if mode == "train" else "./data/original/part_A_final/test_data/images"
    ground_truth = "./data/preprocessed/{0}_density".format(mode) \
        if mode == "train" else "./data/original/part_A_final/test_data/ground_truth_csv"
    data_files = [filename for filename in os.listdir(data_path) \
                 if os.path.isfile(os.path.join(data_path, filename))]
    result = []
    num_files = len(data_files)
    index = 0
    for fname in data_files:
        # load images
        img = skimage.io.imread(os.path.join(data_path, fname)).astype(np.float32)
        if img.shape[-1] == 3:
            img = rgb2gray(img)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht // 4) * 4
        wd_1 = (wd // 4) * 4
        img = skimage.transform.resize(img, (wd_1, ht_1))
        img = np.reshape(img, (wd_1, ht_1, 1))
        img = np.transpose(img, (2, 0, 1))
        # load densities
        den = pd.read_csv(os.path.join(ground_truth, os.path.splitext(fname)[0] + '.csv'),
                          sep=",", header=None).values
        den = den.astype(np.float32, copy=False)
        ht_1 = ht_1 // 4
        wd_1 = wd_1 // 4
        den = skimage.transform.resize(den, (wd_1, ht_1))
        den *= ((wd * ht) // (wd_1 * ht_1))
        index += 1
        # print load speed
        if index % 100 == 0 or index == len(data_files):
            print("load {0}/{1} images ".format(index, num_files))
        result.append([img, den])

    return result


class ShanghaiTechDataset(Dataset):

    def __init__(self, mode="train"):
        self.dataset = get_data(mode=mode)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)