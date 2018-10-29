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


def get_data(mode="train"):
    num_images = 300 if mode=="train" else 182
    # index train_image ground_truth
    train_path = "./data/preprocessed/{0}".format(mode)
    ground_truth = "./data/preprocessed/{0}_density".format(mode)
    result = []
    for i in range(1, num_images + 1):
        for j in range(9):
            image_index = "{0}_{1}.jpg".format(i, j)
            ground_truth_index = "{0}_{1}.npy".format(i, j)
            image_path = os.path.join(train_path, image_index)
            ground_truth_path = os.path.join(ground_truth, ground_truth_index)
            image = skimage.io.imread(image_path)
            density = np.load(ground_truth_path)
            image = np.transpose(image.reshape(image.shape[0], image.shape[1], 1), [2, 0, 1])
            image = torch.from_numpy(image)
            density = torch.from_numpy(density)
            result.append([image, density])

    return result


class ShanghaiTechDataset(Dataset):

    def __init__(self, mode):
        self.dataset = get_data(mode=mode)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
