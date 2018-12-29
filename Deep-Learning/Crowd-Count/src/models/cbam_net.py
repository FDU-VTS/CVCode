# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-12
# ------------------------

import torch.nn as nn
from torchvision import models
from src.utils import utils
import torch


class ChannelAttention(nn.Module):

    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.sig = nn.Sigmoid()

    def forward(self, x):
        _, c, h, w = x.size()
        channel_max = nn.MaxPool2d((h, w))
        channel_avg = nn.AvgPool2d((h, w))
        max_x = channel_max(x)
        avg_x = channel_avg(x)
        final_x = self.sig(max_x + avg_x)
        x = x * final_x
        return x


class SpatialAttention(nn.Module):

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.sig = nn.Sigmoid()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        _, c, h, w = x.size()
        max_x = torch.max(x, 1, keepdim=True)
        avg_x = torch.mean(x, 1, keepdim=True)
        final_x = torch.cat((max_x, avg_x), 1)
        final_x = self.conv2d(final_x)
        final_x = self.sig(final_x)
        x = x * final_x
        return x


class CBAMBlock(nn.Module):

    def __init__(self, in_channels):
        super(CBAMBlock, self).__init__()
        self.in_channels = in_channels
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x_final = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x_final += x
        return x_final


class CBAMNet(nn.Module):

    def __init__(self, load_weights=False):
        super(CBAMNet, self).__init__()
        print("*****init CBAM net*****")
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            utils.weights_normal_init(self)
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_cbam(cfg, in_channels=512, batch_norm=False):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        cbam= CBAMBlock(v)
        layers += [conv2d, cbam]
        in_channels = v
    return nn.Sequential(*layers)
