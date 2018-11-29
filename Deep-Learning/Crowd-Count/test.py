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
warnings.filterwarnings("ignore")
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
models = {
    'mcnn': utils.weights_normal_init(mcnn.MCNN(bn=False), dev=0.01),
    'csr_net': csr_net.CSRNet(),
    'sa_net': sa_net.SANet(input_channels=1, kernel_size=[1, 3, 5, 7], bias=True),
    'tdf_net': utils.weights_normal_init(tdf_net.TDFNet(), dev=0.01),
    'inception': inception.Inception()
}
"""
 load data ->
 init net ->
 backward ->
 test
 zoom size: means reduction rate
 results: mcnn | csr_net | sa_net | tdf_net
 dataset: shtu_dataset | mall_dataset
"""

print("train data loading..........")
mall_data = mall_dataset.MallDataset(img_path="./mall_dataset/frames/", point_path="./mall_dataset/mall_gt.mat",
                                     zoom_size=1)
tech_loader = torch.utils.data.DataLoader(mall_data, batch_size=1, shuffle=True, num_workers=4)
print("test data loading............")
mall_test_data = mall_data
test_loader = torch.utils.data.DataLoader(mall_test_data, batch_size=1, shuffle=False, num_workers=4)
model = models["sa_net"]
# model = nn.DataParallel(model, device_ids=[3])
model_path = "./models/sanet_best_model"
model.load_state_dict(torch.load(model_path))

model.to(DEVICE)
sum_mae = 0.0
sum_mse = 0.0

print("begin")
for i, (input, ground_truth) in enumerate(test_loader):
    input = input.float().to(DEVICE)
    ground_truth = ground_truth.float().to(DEVICE)
    output = model(input)
    mae, mse = utils.get_test_loss(output, ground_truth)
    sum_mae += float(mae)
    sum_mse += float(mse)

print("mae: {mae}, mse: {mse}".format(mae=sum_mae / len(test_loader), mse=math.sqrt(sum_mse / len(test_loader))))