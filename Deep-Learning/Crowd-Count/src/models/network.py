# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-11
# ------------------------
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True):
        super(ConvUnit, self).__init__()
        padding = int((kernel_size - 1)/2)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True),
            nn.ReLU(inplace=True),
        ) if bn else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)

        return x


class FC(nn.Module):

    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)

        return x