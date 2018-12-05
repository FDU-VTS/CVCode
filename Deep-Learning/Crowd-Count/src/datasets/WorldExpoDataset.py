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
import math, os, scipy
import matplotlib.pyplot as plt

def gaussian_filter_density(gt, pts):
    # print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = len(pts)
    if gt_count == 0:
        return density

    # print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[int(round(pt[1])),int(round(pt[0]))] = 1.
        if gt_count > 1:
            # print(round(pt[1]))
            # sigma = (round(pt[1])/50+1)
            sigma = 3
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    # print('done.')
    return density


class WorldExpoDataset(Dataset):
    def __init__(self, img_path, point_path):
        self.img_path = img_path
        self.point_path = point_path
        self.img_list = glob.glob(self.img_path+'*.jpg')

    def __len__(self):
        return  len(self.img_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])
        gray = color.rgb2gray(image)
        points = loadmat(self.point_path+self.img_list[idx].split('/')[3].split('_')[0]+'/'
                         +self.img_list[idx].split('/')[3].replace('.jpg', '.mat'))['point_position']
        density = gaussian_filter_density(gray, points)
        # density = cv2.resize(density, (80, 60), interpolation=cv2.INTER_AREA)

        # numpy_array
        density = torch.tensor(density)
        density = torch.unsqueeze(density, 0)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        # torch.Size([3, 576, 720]) torch.Size([1, 576, 720])
        return image, density

class WorldExpoTestDataset(Dataset):
    def __init__(self, img_path, point_path, type):
        # type 'scene1'|'scene2'|'scene3'|'scene4'|'scene5'
        self.img_path = img_path
        self.point_path = point_path
        self.type = type
        if type is 'scene1':
            self.subdir = '104207/'
        elif type is 'scene2':
            self.subdir = '200608/'
        elif type is 'scene3':
            self.subdir = '200702/'
        elif type is 'scene4':
            self.subdir = '202201/'
        elif type is 'scene5':
            self.subdir = '500717/'
        else:
            return 1
        self.img_list = glob.glob(self.img_path+self.subdir+'*.jpg')

    def __len__(self):
        # print(len(self.point_list))
        return len(self.img_list)
        # print("len", self.point.shape[0]-39)
        # return self.point.shape[0]

    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])

        points = loadmat(self.point_path+self.subdir+self.img_list[idx].split('/')[4].replace('.jpg', '.mat'))['point_position']

        count = [len(points)]
        count = torch.tensor(count, dtype=torch.double)
        # torch.Size([1])

        # numpy_array
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        # torch.Size([3, 576, 720]) torch.Size([1, 576, 720])
        return image, count

def average_scene(scene1, scene2, scene3, scene4, scene5):
    return (scene1 + scene2 + scene3 + scene4 + scene5)/5

if __name__ == "__main__":
    img_path = "./world_expo/test_frame/"
    point_path = './world_expo/test_label/'
    data = WorldExpoTestDataset(img_path, point_path, 'scene1')
    # img_path = "./world_expo/train_frame/"
    # point_path = './world_expo/train_label/'
    # data = WorldExpoDataset(img_path, point_path)
    print(data.__len__())
    a, b = data.__getitem__(30)
    print(a.size(), b.size())
    # a = a.permute(1, 2, 0)
    # b = b.squeeze()
    # a.numpy()
    # b.numpy()
    # plt.imshow(a)
    # plt.show()
    # plt.imshow(b, cmap='hot')
    # plt.show()
    # print(a.size(), b.size())