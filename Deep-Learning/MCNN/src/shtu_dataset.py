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
import torch
import pandas as pd


def get_data(mode="train"):
    num_images = 300 if mode=="train" else 182
    # index train_image ground_truth
    train_path = "./data/preprocessed/{0}".format(mode)
    ground_truth = "./data/preprocessed/{0}_density".format(mode)
    data_files = [filename for filename in os.listdir(train_path) \
                 if os.path.isfile(os.path.join(train_path, filename))]
    result = []
    for fname in data_files:
        img = skimage.io.imread(os.path.join(train_path, fname)).astype(np.float32)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht // 4) * 4
        wd_1 = (wd // 4) * 4
        img = skimage.transform(img, (wd_1, ht_1))
        den = pd.read_csv(os.path.join(ground_truth, os.path.splitext(fname)[0] + '.csv'),
                          sep=",", header=None).values
        den = den.astype(np.float32, copy=False)
        wd_1 = wd_1 // 4
        ht_1 = ht_1 // 4
        den = skimage.transform.resize(den, (wd_1, ht_1))
        den *= ((wd * ht) // wd_1 * ht_1)

        result.append([img, den])

    return result


class ShanghaiTechDataset(Dataset):

    def __init__(self, mode="train"):
        self.dataset = get_data(mode=mode)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

