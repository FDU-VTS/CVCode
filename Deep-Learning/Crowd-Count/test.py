# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils import utils
from src.datasets import mall_dataset, shtu_dataset, shtu_dataset_csr
from src.models import csr_net, sa_net, tdf_net, mcnn, vgg, cbam_net, big_net
import torch
import torch.utils.data
import torch.optim as optim
import warnings
import sys
import math
import numpy as np
from tensorboardX import SummaryWriter
import os
from torchvision import transforms
import torch.nn as nn
import skimage.io
import cv2
import skimage.transform
from skimage.color import grey2rgb
import skimage.filters
import glob
import matplotlib.pyplot as plt
import re
import h5py
from torch.utils.data import Dataset
import matplotlib.cm as cm
from sklearn import preprocessing
warnings.filterwarnings("ignore")
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
"""
 load data ->
 init net ->
 backward ->
 test
 zoom size: means reduction rate
 results: mcnn | csr_net | sa_net | tdf_net
 dataset: shtu_dataset | mall_dataset
"""


class testDataset(Dataset):

    def __init__(self, mode="train", **kwargs):
        self.root = "./data/shtu_dataset/original/part_A_final/train_data/" if mode == "train" else \
                "./data/shtu_dataset/original/part_A_final/test_data/"
        self.paths = glob.glob(self.root + "images/*.jpg")
        if mode == "train":
            self.paths *= 4
        self.transform = kwargs['transform']
        self.length = len(self.paths)
        self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img, den = self.dataset[item]
        img_cp = img
        if self.transform is not None:
            img = self.transform(img)
        return img, den, img_cp

    def load_data(self):
        result = []
        index = 0
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = skimage.io.imread(img_path)
            img = grey2rgb(img)
            gt_file = h5py.File(gt_path)
            den = np.asarray(gt_file['density'])
            h = den.shape[0]
            w = den.shape[1]
            h_trans = h // 8
            w_trans = w // 8
            den = cv2.resize(den, (w_trans, h_trans),
                             interpolation=cv2.INTER_CUBIC) * (h * w) / (h_trans * w_trans)
            result.append([img, den])
            if index % 100 == 99 or index == self.length:
                print("load {0}/{1} images".format(index + 1, self.length))
            index += 1
        return result

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
test_data = testDataset(mode="test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
cbam = cbam_net.CBAMNet()
cbam.load_state_dict(torch.load("./models/cbam_net/best_cbam_net.pkl", map_location="cpu"))
cbam = cbam.eval().to("cpu")
csr = csr_net.CSRNet()
csr.load_state_dict(torch.load("./models/csr_net/best_csr_net.pkl", map_location="cpu"))
csr = csr.eval().to("cpu")
test_loss = utils.LossFunction("test")
for index, (input, ground_truth, input_cp) in enumerate(test_loader):
    input_cp = input_cp.numpy()[0]
    input = input.to(DEVICE)
    ground_truth = ground_truth.float().to(DEVICE)
    cbam_output = cbam(input)[0, 0].detach().numpy()
    csr_output = csr(input)[0, 0].detach().numpy()
    ground_truth = ground_truth[0].detach().numpy()

    weight2 = csr.weight2
    weight2 = weight2[0].detach().numpy()
    weight2 = np.sum(weight2, axis=0)
    weight2 = preprocessing.normalize(weight2, norm='l2')
    weight2 = skimage.filters.gaussian(weight2, sigma=1)

    weight = cbam.get_weight()
    weight = weight[0].detach().numpy()
    print(weight.shape)
    weight = np.sum(weight, axis=0)
    weight = preprocessing.normalize(weight, norm='l2')
    weight = skimage.filters.gaussian(weight, sigma=1)
    print(weight.shape)

    plt.subplot(2, 4, 1)
    plt.imshow(input_cp)
    plt.title("otiginal")
    plt.subplot(2, 4, 2)
    plt.imshow(ground_truth, cmap='RdBu')
    plt.title("gt: " + str(np.sum(ground_truth)))
    plt.subplot(2, 4, 3)
    plt.imshow(csr_output, cmap=cm.jet)
    plt.title("csr: " + str(np.sum(csr_output)))
    plt.subplot(2, 4, 4)
    plt.imshow(cbam_output, cmap=cm.jet)
    plt.title("cbam: " + str(np.sum(cbam_output)))
    plt.subplot(2, 4, 5)
    plt.imshow(weight, cmap='jet')
    plt.subplot(2, 4, 6)
    plt.imshow(weight2, cmap='jet')
    plt.pause(0.5)
