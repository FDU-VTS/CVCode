# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from torch.utils.data import Dataset
import os
import skimage.io
import numpy as np
import torch


def get_train():
    # index train_image ground_truth
    train_path = "./data/preprocessed/train"
    ground_truth = "./data/preprocessed/train_density"
    result = []
    for i in range(1, 301):
        for j in range(9):
            image_index = "{0}_{1}.jpg".format(i, j)
            ground_truth_index = "{0}_{1}.npy".format(i, j)
            image_path = os.path.join(train_path, image_index)
            ground_truth_path = os.path.join(ground_truth, ground_truth_index)
            image = skimage.io.imread(image_path)
            density = np.load(ground_truth_path)
            image = np.transpose(image_path, [2, 0, 1])
            image = torch.from_numpy(image)
            density = torch.from_numpy(density)
            result.append([image, density])

    return result


class ShanghaiTechDataset(Dataset):

    def __init__(self):
        self.dataset = get_train()

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
