# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch.nn as nn
import torch


def get_loss(output, ground_truth):
    number = len(output)
    loss_function = nn.MSELoss()
    loss = 0.0
    for i in range(number):
        output_density = output[i].view(output.size(2), output.size(3))
        ground_truth_density = ground_truth[i]
        loss += loss_function(output_density, ground_truth_density)

    return loss / (2 * number)


def test_loss(output, ground_truth):
    number = len(output)
    mae = 0
    mse = 0
    for i in range(number):
        output_density = output[i].view(output.size(2), output.size(3))
        ground_truth_density = ground_truth[i]
        print(torch.sum(output_density), torch.sum(ground_truth_density))
        diff = abs(torch.sum(output_density) - torch.sum(ground_truth_density))
        mae += diff
        mse += diff ** 2

    return mae / number, torch.sqrt(mse / number)


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

    return model