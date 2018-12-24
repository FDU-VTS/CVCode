# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
"""
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
"""
import torch.nn as nn
import torch
from .network import ConvUnit


class MCNN(nn.Module):

    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        print("*****init MCNN net*****")
        self.column1 = nn.Sequential(
            ConvUnit(3, 16, 9, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(16, 32, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(32, 16, 7, bn=bn),
            ConvUnit(16, 8, 7, bn=bn),
        )

        self.column2 = nn.Sequential(
            ConvUnit(3, 20, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(20, 40, 5, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(40, 20, 5, bn=bn),
            ConvUnit(20, 10, 5, bn=bn),
        )

        self.column3 = nn.Sequential(
            ConvUnit(3, 24, 5, bn=bn),
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

