# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils import utils
from src.datasets import mall_dataset, shtu_dataset, shtu_dataset_test, shtu_dataset_csr
from src.models import csr_net, sa_net, tdf_net, mcnn, aspp, attention_net
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
import re
import h5py
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


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
test_data =shtu_dataset_test.ShanghaiTechDataset(mode="test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
net = csr_net.CSRNet()
# net.load_state_dict(torch.load("./models/csr_net/best_csr_net_4.pkl"))
net.load_state_dict(torch.load("./models/PartAmodel_best.pth.tar")["state_dict"])
net = net.to(DEVICE)
net = net.eval()
sum_mae = 0.0
sum_mse = 0.0
test_loss = utils.LossFunction("test")
save_path = "./results/"
for index, (input, ground_truth, input_cp) in enumerate(test_loader):
    input = input.to(DEVICE)
    ground_truth = ground_truth.float().to(DEVICE)
    output = net(input)
    mae, mse = test_loss(output, ground_truth)
    print("mae:{0}, mse:{1}".format(mae, math.sqrt(mse)))
    sum_mae += float(mae)
    sum_mse += float(mse)
    # ground_truth = ground_truth[0].detach().numpy()
    # output = output[0, 0].detach().numpy()
    # plt.subplot(1, 3, 1)
    # plt.imshow(input_cp)
    # plt.title(str(mae.detach().numpy()))
    # plt.subplot(1, 3, 2)
    # plt.imshow(ground_truth, cmap='hot')
    # plt.title("gt: " + str(np.sum(ground_truth)))
    # plt.subplot(1, 3, 3)
    # plt.imshow(output, cmap='hot')
    # plt.title("out: " + str(np.sum(output)))
    # plt.pause(0.5)
print("mae:%.1f, mse:%.1f" % (sum_mae / len(test_loader), math.sqrt(sum_mse / len(test_loader))))



# image_path = "./data/shtu_dataset/preprocessed/test/"
# image_paths = glob.glob(os.path.join(image_path, "*.jpg"))
# for image_path in image_paths:
#     p = re.compile("test/(.*).jpg")
#     name = str(p.findall(image_path)[0])
#     gt_2 = h5py.File(os.path.join("./data/shtu_dataset/original/part_A_final/test_data/ground_truth",
#                                   name + ".h5"))
#     gt_2 = np.asarray(gt_2["density"])
#     gt_path = image_path.replace("test", "test_density").replace("jpg", "npy")
#     img = skimage.io.imread(image_path)
#     gt = np.load(gt_path)
#     plt.subplot(1, 3, 1)
#     plt.imshow(img)
#     plt.subplot(1, 3, 2)
#     plt.imshow(gt, cmap="hot")
#     plt.subplot(1, 3, 3)
#     plt.imshow(gt_2, cmap="hot")
#     plt.pause(2)