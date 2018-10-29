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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    print("data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=2)
    print("init net............")
    net = mcnn.MCNN().to(DEVICE)
    net.load_state_dict(torch.load("./model/mcnn.pkl"), strict=False)
    i = 0
    for input, ground_truth in iter(test_loader):
        print(i)
        i += 1
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        loss, people_number, ground_number = utils.test_loss(output, ground_truth)
        print(loss, people_number, ground_number)
        result = output[0].cpu()
        result = result.detach().numpy()
        result = np.transpose(result, [1, 2, 0])
        result = result.reshape(result.shape[0], result.shape[1])
        np.save("./data/result/result/{0}.npy".format(i), result)


if __name__ == "__main__":
    test()
