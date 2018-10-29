# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

import mcnn
import shtu_dataset
import torch.utils.data
import utils
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    print("data loading............")
    test_data = shtu_dataset.ShanghaiTechDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=2)
    print("init net............")
    net = mcnn.MCNN().to(DEVICE)
    net.load_state_dict(torch.load("./model/mcnn.pkl"))
    for input, ground_truth in iter(test_loader):

        input = input.float().to(DEVICE)
        ground_truth = ground_truth.float().to(DEVICE)
        output = net(input)
        loss, people_number, ground_number = utils.get_loss(output, ground_truth)
        print("loss: {loss}, people_number: {people_number}, ground_number: {ground_number}".format(loss=loss,
                                                                                                    people_number=people_number,
                                                                                                    ground_number=ground_number))


if __name__ == "__main__":
    test()