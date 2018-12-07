# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils import utils
from src.datasets import mall_dataset, shtu_dataset
from src.models import csr_net, sa_net, tdf_net, mcnn, inception
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
import glob
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
"""
 load data ->
 init net ->
 backward ->
 test
 zoom size: means reduction rate
 results: mcnn | csr_net | sa_net | tdf_net
 dataset: shtu_dataset | mall_dataset
"""

# path = "./data/shtu_dataset/original/part_A_final/train_data/test/IMG_1.jpg"

paths = ["./data/shtu_dataset/preprocessed/train_density/", "./data/shtu_dataset/preprocessed/test_density/", "./data/shtu_dataset/preprocessed/test/"]
images_path = paths[2]
gt_path = paths[0]
images_path = glob.glob(os.path.join(images_path, "*.jpg"))
gt_path = glob.glob(os.path.join(gt_path, "*.npy"))
for image_path in images_path:
    gt = image_path.replace('test', 'test_density').replace('jpg', 'npy')
    image = skimage.io.imread(image_path)
    gt_image = np.load(gt)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(gt_image)
    plt.pause(2)