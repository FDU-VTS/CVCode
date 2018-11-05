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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    print("init net............")
    net = mcnn.MCNN().eval().to(DEVICE)
    net.load_state_dict(torch.load("./model/mcnn.pkl"), strict=False)
    print("start to predict...........")
    sum_mae = 0.0
    sum_mse = 0.0
    for input, ground_truth in iter(test_loader):
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        mae, mse = utils.get_test_loss(output, ground_truth)
        sum_mae += float(mae)
        sum_mse += float(mse)
    print("best_mae:%.1f, best_mse:%.1f" % (sum_mae / len(test_loader), math.sqrt(sum_mse / len(test_loader))))


if __name__ == "__main__":
    test()
