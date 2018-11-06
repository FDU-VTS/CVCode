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


def test():


    net = mcnn.MCNN()
    net.to("cuda")
    net.load_staste_dict(torch.load("./model/mcnn.pkl"), strict=False)
    net = net.to("cpu")

    print("start to predict...........")
    sum_mae = 0.0
    sum_mse = 0.0
    a = torch.zeros(1,1,1,1)
    a = a.float()
    net(a)


if __name__ == "__main__":
    test()
