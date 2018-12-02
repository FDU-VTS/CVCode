# --------------------------------------------------------
# SANet
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import numpy as np
from scipy.io import loadmat
import cv2
import torch
from torch.utils.data import Dataset
from skimage import io, color
import math, os
import skimage.transform


def gaussian_kernel(image, points):
    image_density = np.zeros(image.shape)
    h, w = image_density.shape

    if len(points) == 0:
        return image_density

    if len(points) == 1:
        x1 = np.max(0, np.min(w, round(points[0, 0])))
        y1 = np.max(0, np.min(h, round(points[0, 1])))
        image_density[y1, x1] = 255
        return image_density

    for j in range(len(points)):
        f_sz = 15
        sigma = 4.0
        kx = cv2.getGaussianKernel(f_sz, sigma=sigma)
        ky = cv2.getGaussianKernel(f_sz, sigma=sigma)
        H = np.multiply(kx, ky.T)
        # convert x, y to int
        x = min(w, max(0, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(0, abs(int(math.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        gap = int(math.floor(f_sz / 2))
        x1 = x - gap if x - gap > 0 else 0
        x2 = x + gap if x + gap < w else w - 1
        y1 = y - gap if y - gap > 0 else 0
        y2 = y + gap if y + gap < h else h - 1
        # generate 2d gaussian kernel
        kx = cv2.getGaussianKernel(y2 - y1 + 1, sigma=sigma)
        ky = cv2.getGaussianKernel(x2 - x1 + 1, sigma=sigma)
        gaussian = np.multiply(kx, ky.T)

        image_density[y1:y2 + 1, x1:x2 + 1] = image_density[y1:y2 + 1, x1:x2 + 1] + gaussian

    return image_density


class MallDataset(Dataset):
    def __init__(self, img_path, point_path, zoom_size=1):
        self.img_path = img_path
        self.point_path = point_path
        ground_truth__dict = loadmat(point_path)
        self.point = ground_truth__dict['frame'][0]
        self.zoom_size = zoom_size
        self.num = self.point.shape[0]

    def __len__(self):
        return self.num
        # return 60

    def __getitem__(self, idx):
        img = io.imread(self.img_path + 'seq_00' + str(idx+1).zfill(4) + '.jpg')
        img = color.rgb2gray(img)
        den = gaussian_kernel(img, self.point[idx]['loc'][0][0])
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht // 8) * 8
        wd_1 = (wd // 8) * 8
        img = cv2.resize(img, (wd_1, ht_1), interpolation=cv2.INTER_AREA)
        img = img[:, :, np.newaxis]
        img = np.transpose(img, (2, 0, 1))
        ht_1 = ht_1 // self.zoom_size
        wd_1 = wd_1 // self.zoom_size
        den = cv2.resize(den, (wd_1, ht_1), interpolation=cv2.INTER_AREA)
        den *= ((wd * ht) // (wd_1 * ht_1))

        return img, den


class MallDatasetTest(Dataset):
    def __init__(self, img_path, point_path):
        self.img_path = img_path
        self.point_path = point_path
        ground_truth__dict = loadmat(point_path)
        self.count = ground_truth__dict['count']

    def __len__(self):
        return self.count.shape[0]

    def __getitem__(self, idx):
        image = io.imread(self.img_path + 'seq_00' + str(idx+1).zfill(4) + '.jpg')
        gray = color.rgb2gray(image)
        count = self.count[idx]
        gray = torch.tensor(gray)
        gray = torch.unsqueeze(gray, 0)
        count = torch.tensor(count, dtype=torch.double)
        return gray, count
