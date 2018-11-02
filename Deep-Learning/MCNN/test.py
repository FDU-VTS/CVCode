# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

from src import shtu_dataset, utils, mcnn
import torch.utils.data
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    print("data loading............")
    test_data = shtu_dataset.ShanghaiTechTestDataset()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    print("init net............")
    net = mcnn.MCNN().eval().to(DEVICE)
    net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.load_state_dict(torch.load("./model/mcnn.pkl"), strict=False)
    i = 0
    sum_mae = 0.0
    sum_mse = 0.0
    print("start to predict...........")
    for input, ground_truth in iter(test_loader):
        i += 1
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        mae, mse = utils.test_loss(output, ground_truth)
        sum_mae += float(mae)
        sum_mse += float(mse)

        if i % 60 == 59:
            print("mae: %f, mse: %f" %(mae / 20, mse / 20))
            sum_mae = 0.0
            sum_mse = 0.0


if __name__ == "__main__":
    test()
