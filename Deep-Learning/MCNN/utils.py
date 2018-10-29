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

    return loss / number


def test_loss(output, ground_truth):
    number = len(output)
    loss_function = nn.MSELoss(size_average=False)
    loss = 0.0
    # people_number: number of people after net
    # ground_number: number of people via gaussian kernel
    people_number = 0.0
    ground_number = 0.0
    for i in range(number):
        output_density = output[i].view(output.size(2), output.size(3))
        ground_truth_density = ground_truth[i]
        loss += loss_function(output_density, ground_truth_density)
        people_number += torch.sum(ground_truth_density)
        ground_number += torch.sum(output_density)

    return loss, torch.round(people_number), torch.round(ground_number)
