# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

import mcnn
import shtu_dataset
import torch.utils.data
import utils
import numpy as np
import skimage.io
import warnings
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    print("data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=2)
    print("init net............")
    net = mcnn.MCNN().to(DEVICE)
    net.load_state_dict(torch.load("./model/mcnn.pkl"), strict=False)
    i = 0
    sum_mae = 0.0
    sum_mse = 0.0
    for input, ground_truth in iter(test_loader):
        
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        mae, mse = utils.test_loss(output, ground_truth)
        sum_mae += mae
        sum_mse += mse

        if i % 50 == 49:
            print("mae: ", sum_mae / 50)
            print("mse: ", sum_mse / 50)
            sum_mae = 0.0
            sum_mse = 0.0


if __name__ == "__main__":
    test()
