# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-11
# ------------------------

import torch.nn as nn
from torchvision import models
from src.utils import utils


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, in_channels=1)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(512, 1, kernel_size=1)
        if not load_weights:
            utils.weights_normal_init(self)
            mod = models.vgg16(pretrained=True)
            pretrained_dict = mod.state_dict()
            frontend_dict = self.frontend.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in frontend_dict}
            frontend_dict.update(pretrained_dict)
            self.frontend.load_state_dict(frontend_dict)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


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
