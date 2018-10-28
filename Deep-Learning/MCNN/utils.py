# -*- coding: utf-8 -*-
import torch.nn as nn


def get_loss(output, ground_truth):
    output = output
    ground_truth = ground_truth
    number = len(output)
    loss_function = nn.MSELoss(size_average=True)
    loss = 0.0
    for i in range(number):
        output_density = output[i].view(output.size(2), output.size(3))
        ground_truth_density = ground_truth[i]
        loss += loss_function(output_density, ground_truth_density)

    return loss / number
