# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

import torch
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False):
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


class MCNN(nn.Module):

    def __init__(self, bn=True):
        super(MCNN, self).__init__()

        self.column1 = nn.Sequential(
            ConvUnit(1, 16, 9, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(16, 32, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(32, 16, 7, bn=bn),
            ConvUnit(16, 8, 7, bn=bn),
        )

        self.column2 = nn.Sequential(
            ConvUnit(1, 20, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(20, 40, 5, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(40, 20, 5, bn=bn),
            ConvUnit(20, 10, 5, bn=bn),
        )

        self.column3 = nn.Sequential(
            ConvUnit(1, 24, 5, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(24, 48, 3, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(48, 24, 3, bn=bn),
            ConvUnit(24, 12, 3, bn=bn),
        )

        self.merge = nn.Sequential(
            ConvUnit(30, 1, 1, bn=bn)
        )

    def forward(self, x):
        x1 = self.column1(x)
        x2 = self.column2(x)
        x3 = self.column3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.merge(x)

        return x
