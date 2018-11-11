# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
# csr_net: 121.7, 177.4
# mcnn: 141.1, 213.2
from src import shtu_dataset, utils, mcnn, csr_net, sa_net
import torch
import torch.utils.data
import torch.optim as optim
import warnings
import sys
import math
import numpy as np
import os
warnings.filterwarnings("ignore")
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
learning_rate = 0.00001
save_path = "./model/mcnn.pkl"


def train(zoom_size=4, model="mcnn"):
    print("train data loading..........")
    shanghaitech_dataset = shtu_dataset.ShanghaiTechDataset(mode="train", zoom_size=zoom_size)
    tech_loader = torch.utils.data.DataLoader(shanghaitech_dataset, batch_size=1, shuffle=True, num_workers=8)
    print("test data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    print("init net...........")
    if model == "mcnn":
        net = mcnn.MCNN()
        net = utils.weights_normal_init(net, dev=0.01)
    elif model == "csr_net":
        net = csr_net.CSRNet()
    elif model == "sa_net":
        net = sa_net.SANet(input_channels=1, kernel_size=[1, 3, 5, 7], bias=True)
    net = net.train().to(DEVICE)
    print("init optimizer..........")
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr=learning_rate)
    print("start to train net.....")
    sum_loss = 0
    step = 0
    result = []
    min_mae = sys.maxsize
    # for each 2 epochs in 2000 get and model to test
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
            step += 1
            if step % 1000 == 0:
                print("{0} patches are done, loss: ".format(step), sum_loss / 500)
                sum_loss = 0

        if epoch % 2 == 0:
            sum_mae = 0.0
            sum_mse = 0.0
            for input, ground_truth in iter(test_loader):
                input = input.float().to(DEVICE)
                ground_truth = ground_truth.float().to(DEVICE)
                output = net(input)
                mae, mse = utils.get_test_loss(output, ground_truth)
                sum_mae += float(mae)
                sum_mse += float(mse)
            if sum_mae / len(test_loader) < min_mae:
                min_mae = sum_mae / len(test_loader)
                min_mse = sum_mse / len(test_loader)
                result.append([min_mae, math.sqrt(min_mse)])
                # torch.save(net.state_dict(), "./model/mcnn-11-11.pkl")
            print("best_mae:%.1f, best_mse:%.1f" % (min_mae, math.sqrt(min_mse)))
        step = 0
    result = np.asarray(result)
    try:
        np.save("./model/mcnn-bn.npy", result)
    except IOError:
        os.mkdir("./model")
        np.save("./model/mcnn-bn.npy", result)
    print("save successful!")


if __name__ == "__main__":
    print("start....")
    model = str(sys.argv[1])
    zoom_size = int(sys.argv[2])
    print("model: {0}, zoom_size: {1}".format(model, zoom_size))
    train(zoom_size=zoom_size, model=model)