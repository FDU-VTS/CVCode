# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils import utils
from src.datasets import mall_dataset, shtu_dataset
from src.models import csr_net, sa_net, tdf_net, mcnn, inception, aspp
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


net = aspp.ASPP()
a = torch.zeros(1, 3, 800, 800)
b = net(a)