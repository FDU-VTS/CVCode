# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as F


class SPPLayer(nn.Module):

    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            tensor = nn.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)

        return x


class DetectionNetSPP(nn.Module):

    def __init__(self, spp_level=3):
        super(DetectionNetSPP, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        print(self.num_grids)

        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )

        self.spp_level = SPPLayer(spp_level)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_grids*128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.conv_model(x)
        x = self.spp_level(x)
        x = self.classifier(x)

        return x