# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

from src import shtu_dataset, utils, mcnn
import torch.utils.data
import warnings
import math
import torch.nn as nn
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    print("data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    print("init net............")
    net = mcnn.MCNN().eval().to(DEVICE)
    net.load_state_dict(torch.load("./model/mcnn.pkl"), strict=False)
    i = 0
    sum_mae = 0.0
    sum_mse = 0.0
    print("start to predict...........")

    for input, ground_truth in iter(test_loader):
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        mae, mse = utils.get_test_loss(output, ground_truth)
        sum_mae += float(mae)
        sum_mse += float(mse)
        i += 1
        if i % 100 == 0:
            print("mae:%.1f, mse:%.1f" % (sum_mae / 100, math.sqrt(sum_mse / 100)))
            sum_mae = 0.0
            sum_mse = 0.0


if __name__ == "__main__":
    test()
