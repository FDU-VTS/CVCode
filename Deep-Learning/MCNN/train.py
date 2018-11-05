# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src import shtu_dataset, utils, mcnn
import torch
import torch.utils.data
import torch.optim as optim
import warnings
import sys
import math
import numpy as np
import os
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 0.00001
save_path = "./model/mcnn.pkl"


def train():
    print("train data loading..........")
    shanghaitech_dataset = shtu_dataset.ShanghaiTechDataset(mode="train")
    tech_loader = torch.utils.data.DataLoader(shanghaitech_dataset, batch_size=1, shuffle=True, num_workers=8)
    print("test data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    print("init net...........")
    net = mcnn.MCNN().train().to(DEVICE)
    net = utils.weights_normal_init(net, dev=0.01)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
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
            if step % 500 == 0:
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
                torch.save(net.state_dict(), "./model/mcnn.pkl")
            print("best_mae:%.1f, best_mse:%.1f" % (min_mae, math.sqrt(min_mse)))
        step = 0
    result = np.asarray(result)
    try:
        np.save("./model/result.npy")
    except IOError:
        os.mkdir("./model")
        np.save("./model/result.npy")


if __name__ == "__main__":
    train()