# --------------------------------------------------------
# WorldExpo Dataset
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import numpy as np
from scipy.io import loadmat
import glob
import cv2
import torch
from torch.utils.data import Dataset
from skimage import io, color, transform
import math, os
import matplotlib.pyplot as plt


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
        sigma = 5
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


class WorldExpoDataset(Dataset):
    def __init__(self, img_path, point_path):
        self.img_path = img_path
        self.point_path = point_path
        self.img_list = glob.glob(self.img_path+'*.jpg')

        # ground_truth__dict = loadmat(point_path)
        # self.point = ground_truth__dict['frame'][0]

    def __len__(self):
        print(len(self.img_list))
        print(len(self.point_list))
        # print(self.img_list)
        # print("len", self.point.shape[0]-39)
        # return self.point.shape[0]

    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])
        gray = color.rgb2gray(image)
        points = loadmat(self.point_path+self.img_list[idx].split('/')[3].split('_')[0]+'/'
                         +self.img_list[idx].split('/')[3].replace('.jpg', '.mat'))['point_position']
        density = gaussian_kernel(gray, points)
        # density = cv2.resize(density, (80, 60), interpolation=cv2.INTER_AREA)

        # numpy_array
        density = torch.tensor(density)
        density = torch.unsqueeze(density, 0)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        # torch.Size([3, 576, 720]) torch.Size([1, 576, 720])
        return image, density

if __name__ == "__main__":
    img_path = "./world_expo/train_frame/"
    point_path = './world_expo/train_label/'
    data = WorldExpoDataset(img_path, point_path)
    a, b = data.__getitem__(3)
    print(a.size(), b.size())