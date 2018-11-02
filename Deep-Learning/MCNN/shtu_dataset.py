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
            # convert image and image density
            h, w = image.shape
            new_h = (h // 4) * 4
            new_w = (w // 4) * 4
            image = skimage.transform.resize(image, [new_h, new_w])
            density = skimage.transform.resize(density, [new_h // 4, new_w // 4])
            density *= (h * w) / (new_h * new_w / 16)
            image = np.transpose(image.reshape(image.shape[0], image.shape[1], 1), [2, 0, 1])
            image = torch.from_numpy(image)
            density = torch.from_numpy(density)
            result.append([image, density])

    return result


def get_test_data(part="A"):
    num_images = 182 if part == "A" else 316
    test_path = "./data/original/part_{part}_final/test_data/images/".format(part=part)
    ground_truth = "./data/preprocessed/test_density/"
    result = []
    for i in range(1, num_images + 1):
        image_index = "IMG_{index}.jpg".format(index=i)
        ground_truth_index = "{index}.npy".format(index=i)
        image_path = os.path.join(test_path, image_index)
        ground_truth_path = os.path.join(ground_truth, ground_truth_index)
        image = skimage.io.imread(image_path)
        density = np.load(ground_truth_path)
        if image.shape[-1] == 3:
            image = rgb2gray(image)
        h, w = image.shape
        new_h = (h // 4) * 4
        new_w = (w // 4) * 4
        image = skimage.transform.resize(image, [new_h, new_w])
        density = skimage.transform.resize(density, [new_h // 4, new_w // 4])
        density *= (h * w) / (new_h * new_w / 16)
        image = np.transpose(image.reshape(image.shape[0], image.shape[1], 1), [2, 0, 1])
        image = torch.from_numpy(image)
        density = torch.from_numpy(density)
        result.append([image, density])

    return result


class ShanghaiTechDataset(Dataset):

    def __init__(self, mode="train"):
        self.dataset = get_data(mode=mode)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


class ShanghaiTechTestDataset(Dataset):

    def __init__(self, part="A"):
        self.dataset = get_test_data(part)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)