# --------------------------------------------------------
# WorldExpo Dataset
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import numpy as np
from scipy.io import loadmat
import glob
import torch
import h5py
from torch.utils.data import Dataset
from skimage import io, color, transform
import math, os, scipy


def gaussian_filter_density(gt, pts):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = len(pts)
    if gt_count == 0:
        return density
    # print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[min(int(round(pt[1])), gt.shape[0]-1), min(int(round(pt[0])), gt.shape[1]-1)] = 1.
        if gt_count > 1:
            sigma = 3
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


class WorldExpoDataset(Dataset):
    def __init__(self, img_path, point_path):
        self.img_path = img_path
        self.point_path = point_path
        wrong_data = ['200778_C10-03-S20100717083000000E20100717233000000_4_clip1_2.jpg',
                      '200778_C10-03-S20100717083000000E20100717233000000_4_clip1_3.jpg',
                      '200778_C10-03-S20100717083000000E20100717233000000_clip1_3.jpg',
                      '500674_E05-03-S20100717083000000E20100717233000000_5_clip1_2.jpg',
                      '600079_E06-02-S20100717083000000E20100717233000000_7_clip1_2.jpg']
        for i in range(len(wrong_data)):
            if os.path.exists(self.img_path + wrong_data[i]):
                os.remove(self.img_path + wrong_data[i])
        self.img_list = glob.glob(self.img_path+'*.jpg')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])
        gray = color.rgb2gray(image)
        read_path = self.point_path+self.img_list[idx].split('/')[-1][:6]+'/'+self.img_list[idx].split('/')[-1].replace('.jpg', '.mat')
        input = open(read_path, 'rb')
        check = float(str(input.read(10)).split('\'')[1].split(' ')[1])
        input.close()
        if check == 7.3:
            points = np.array(h5py.File(read_path, 'r')['point_position'])
            points = points.transpose()
        else:
            points = loadmat(read_path)['point_position']

        density = gaussian_filter_density(gray, points)

        # numpy_array
        density = torch.tensor(density)
        density = torch.unsqueeze(density, 0)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
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
        wrong_data = ['104207/104207_1-04-S20100821071000000E20100821120000000_034550.jpg']
        for i in range(len(wrong_data)):
            if os.path.exists(self.img_path + wrong_data[i]):
                os.remove(self.img_path + wrong_data[i])
        self.img_list = glob.glob(self.img_path+self.subdir+'*.jpg')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])
        read_path = self.point_path+self.subdir+self.img_list[idx].split('/')[-1].replace('.jpg', '.mat')
        input = open(read_path, 'rb')
        check = float(str(input.read(10)).split('\'')[1].split(' ')[1])
        input.close()
        if check == 7.3:
            points = np.array(h5py.File(read_path, 'r')['point_position'])
            points = points.transpose()
        else:
            points = loadmat(read_path)['point_position']

        count = [len(points)]
        count = torch.tensor(count, dtype=torch.double)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        return image, count


def average_scene(scene1, scene2, scene3, scene4, scene5):
    return (scene1 + scene2 + scene3 + scene4 + scene5)/5


if __name__ == "__main__":
    img_path = "./world_expo/train_frame/"
    point_path = './world_expo/train_label/'
    data = WorldExpoDataset(img_path, point_path)