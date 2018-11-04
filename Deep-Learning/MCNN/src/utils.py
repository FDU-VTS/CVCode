# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch.nn as nn
import torch
import numpy as np


def get_loss(output, ground_truth):

    loss_function = nn.MSELoss()
    output_density = output[0].view(output.size(2), output.size(3))
    ground_truth_density = ground_truth[0]
    loss = loss_function(output_density, ground_truth_density)

    return loss / 2


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
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

    return model