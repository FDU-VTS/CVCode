# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch.nn as nn
import torch


def get_loss(output, ground_truth):
    number = len(output)
<<<<<<< HEAD
    loss_function = nn.MSELoss(size_average=False, reduce=True)
=======
    loss_function = nn.MSELoss(size_average=False)
>>>>>>> 1302afd24652a7e144eea15e6cb89ff2764f3a00
    loss = 0.0
    for i in range(number):
        output_density = output[i].view(output.size(2), output.size(3))
        ground_truth_density = ground_truth[i]
        loss += loss_function(output_density, ground_truth_density)

    return loss / number


def test_loss(output, ground_truth):
    number = len(output)
    mae = 0
    mse = 0
    for i in range(number):
        output_density = output[i].view(output.size(2), output.size(3))
        ground_truth_density = ground_truth[i]
        diff = output_density - ground_truth_density
        mae += abs(torch.sum(diff))
        mse += torch.sum(diff ** 2)

    return mae / number, mse / number
