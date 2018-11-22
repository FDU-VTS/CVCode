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
import torch.nn as nn
import warnings
import sys
import math
import numpy as np
from tensorboardX import SummaryWriter
import torchvision
warnings.filterwarnings("ignore")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-5
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


def train(zoom_size=4, model="mcnn", dataset="shtu_dataset"):
    """

    :type zoom_size: int
    :type model: str
    :type dataset: str

    """
    # load data
    if dataset == "shtu_dataset":
        print("train data loading..........")
        shanghaitech_dataset = shtu_dataset.ShanghaiTechDataset(mode="train", zoom_size=zoom_size)
        tech_loader = torch.utils.data.DataLoader(shanghaitech_dataset, batch_size=1, shuffle=True, num_workers=1)
        print("test data loading............")
        test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    elif dataset == "mall_dataset":
        print("train data loading..........")
        mall_data = mall_dataset.MallDataset(img_path="./mall_dataset/frames/", point_path="./mall_dataset/mall_gt.mat", zoom_size=zoom_size)
        tech_loader = torch.utils.data.DataLoader(mall_data, batch_size=1, shuffle=True, num_workers=4)
        print("test data loading............")
        mall_test_data = mall_data
        test_loader = torch.utils.data.DataLoader(mall_test_data, batch_size=1, shuffle=False, num_workers=4)
    print("init net...........")
    net = models[model]
    net = net.train().to(DEVICE)
    net = nn.DataParallel(net, device_ids=[0, 1])
    print("init optimizer..........")
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr=learning_rate)
    print("start to train net.....")
    sum_loss = 0.0
    epoch_index = 0
    i = 0
    min_mae = sys.maxsize
    model_dir = model + "_" + dataset
    writer = SummaryWriter('runs/'+model_dir)
    # for each 2 epochs in 2000 get and results to test
    # and keep the best one
    for epoch in range(2000):
        for input, ground_truth in iter(tech_loader):
            input = input.float().to(DEVICE)
            ground_truth = ground_truth.float().to(DEVICE)
            output = net(input)
            loss = utils.get_loss(output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
            if i % 100 == 99:
                print("{0} steps' loss is {1}".format(i + 1, sum_loss / 100))
                sum_loss = 0
            i += 1

        if epoch % 2 == 0:
            test_net = net
            test_net = test_net.to("cuda:1")
            sum_mae = 0.0
            sum_mse = 0.0
            for input, ground_truth in iter(test_loader):
                input = input.float().to("cuda:1")
                ground_truth = ground_truth.float().to("cuda:1")
                output = test_net(input)
                mae, mse = utils.get_test_loss(output, ground_truth)
                sum_mae += float(mae)
                sum_mse += float(mse)
            if sum_mae / len(test_loader) < min_mae:
                min_mae = sum_mae / len(test_loader)
                min_mse = sum_mse / len(test_loader)
            print("best_mae:%.1f, best_mse:%.1f" % (min_mae, math.sqrt(min_mse)))
            print("{0} epoches / 2000 epoches are done".format(epoch_index)),
        writer.add_scalar(model_dir + "/loss", np.asarray(sum_loss / len(test_loader), dtype=np.float32), epoch_index)
        writer.add_scalar(model_dir + "/mae", np.asarray(min_mae), epoch_index)
        writer.add_scalar(model_dir + "/mse", np.asarray(min_mse), epoch_index)
        print("{0} patches are done, loss: ".format(epoch_index), sum_loss / len(test_loader))
        epoch_index += 1
        sum_loss = 0.0

    writer.close()


if __name__ == "__main__":
    # args: zoom_size, model, dataset
    print("start....")
    model = str(sys.argv[1])
    zoom_size = int(sys.argv[2])
    dataset = str(sys.argv[3])
    print("results: {0}, zoom_size: {1}".format(model, zoom_size))
    train(zoom_size=zoom_size, model=model, dataset=dataset)