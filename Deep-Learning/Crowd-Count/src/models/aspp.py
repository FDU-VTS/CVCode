# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-11
# ------------------------

import torch.nn as nn
from torchvision import models
from src.utils import utils
import torch


class ASPP(nn.Module):
    def __init__(self, load_weights=False):
        super(ASPP, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            utils.weights_normal_init(self)
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


class Fusion(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilate1_1 = self._dilate_conv(1)
        self.dilate2_1 = self._dilate_conv(2)
        self.dilate3_1 = self._dilate_conv(3)
        self.dilate4_1 = self._dilate_conv(4)

    def forward(self, x):
        x1 = self.dilate1_1(x)
        x2 = self.dilate2_1(x)
        x3 = self.dilate3_1(x)
        return torch.cat((x1, x2, x3, x4), 1)

    def _dilate_conv(self, rate):
        out = self.out_channels // 4
        return nn.Conv2d(in_channels=self.in_channels, out_channels=out, kernel_size=3, padding=rate, dilation=rate)


def make_fusion(cfg, in_channels=512):
    layers = []
    for v in cfg:
        fusion = Fusion(in_channels, v)
        layers += [fusion, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)