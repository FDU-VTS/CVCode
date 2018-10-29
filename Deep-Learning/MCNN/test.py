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
        
        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        loss, people_number, ground_number = utils.test_loss(output, ground_truth)
        result = output[0].cpu()
        result = result.detach().numpy()
        result = np.transpose(result, [1, 2, 0])
        print(result.shape)
        skimage.io.imsave("./data/result/result/{0}.jpg".format(i), result)
        i += 1
        print("loss: {loss}, people_number: {people_number}, ground_number: {ground_number}".format(loss=loss,
                                                                                                    people_number=people_number,
                                                                                                    ground_number=ground_number))


if __name__ == "__main__":
    test()
