# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils.ssim import SSIM
import torch.nn as nn
import torch


def sa_loss(output, ground_truth):
    loss = get_loss(output, ground_truth)
    ground_truth = torch.unsqueeze(ground_truth, 0)
    ssim_loss = SSIM()
    loss += 0.001 * (1 - ssim_loss(output, ground_truth))
    return loss


# loss between density and ground truth
def get_loss(output, ground_truth):
    number = len(output)
    loss_function = nn.MSELoss()
    loss = 0.0
    for i in range(number):
        output_density = output[i]
        ground_truth_density = ground_truth[i]
        loss += loss_function(output_density, ground_truth_density)

    return loss / (2 * number)


def get_test_loss(output, ground_truth):
    output_density = output[0].view(output.size(2), output.size(3))
    ground_truth_density = ground_truth[0]
    sum_output = torch.sum(output_density)
    sum_gt = torch.sum(ground_truth_density)
    mae = abs(sum_output - sum_gt)
    mse = (sum_output - sum_gt) * (sum_output - sum_gt)

    return mae, mse


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return model
