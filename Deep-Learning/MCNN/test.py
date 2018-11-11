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
import warnings
import math
warnings.filterwarnings("ignore")
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def test():
    print("test data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    print("init net...........")
    net = mcnn.CrowdCounter()
    utils.load_net("./model/mcnn_shtechA_660.h5", net)
    net = net.to(DEVICE)
    # for each 2 epochs in 2000 get and model to test
    # and keep the best one
    sum_mae = 0.0
    sum_mse = 0.0
    i = 0
    for input, ground_truth in iter(test_loader):
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        mae, mse = utils.get_test_loss(output, ground_truth)
        sum_mae += float(mae)
        sum_mse += float(mse)
        i += 1
        if i % 50 == 0:
            print("{0} images".format(i))
    print("best_mae:%.1f, best_mse:%.1f" % (sum_mae / len(test_loader), math.sqrt(sum_mse/len(test_loader))))


if __name__ == "__main__":
    print("start....")
    test()