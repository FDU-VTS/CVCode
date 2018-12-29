# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-11
# ------------------------
import torch.nn as nn
import math
import torch
from torchvision import models
from src.utils import utils
from .network import FC


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
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


class SPPNet(nn.Module):

    def __init__(self):
        super(SPPNet, self).__init__()

    def forward(self, features):
        # get 3*3 2*2 1*1 spatial pyramid pooling
        # features: (1, h, w)
        b, c, h, w = features.size()
        result = torch.tensor([])
        for k in [1, 2, 3]:
            gap_h = h // k
            gap_w = w // k
            for i in range(k):
                for j in range(k):
                    x = torch.max(features[:, :, gap_h * i:gap_h * (i + 1), gap_w * j:gap_w * (j + 1)])
                    result = torch.cat((result, torch.tensor([x])), 0)
        result = torch.unsqueeze(result, 0)
        return result


class SRN(nn.Module):

    def __init__(self):
        super(SRN, self).__init__()
        self.fc = FC(14 * 3, 3)
        self.spp_net = SPPNet()

    def forward(self, x1, x2, x3):
        x1 = self.spp_net(x1)
        x2 = self.spp_net(x2)
        x3 = self.spp_net(x3)
        x = torch.cat((x1, x2, x3), 0)
        x = x.view(1, -1)
        x = x.to("cuda:0")
        x = self.fc(x)
        return x


class PaDNet(nn.Module):

    def __init__(self):
        super(PaDNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend_1 = make_layers(self.backend_feat, in_channels=512, d_rate=1)
        self.backend_2 = make_layers(self.backend_feat, in_channels=512, d_rate=2)
        self.backend_3 = make_layers(self.backend_feat, in_channels=512, d_rate=3)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.out_layer = nn.Conv2d(3 ,1, kernel_size=1)
        self.srn = SRN()
        mod = models.vgg16(pretrained=True)
        utils.weights_normal_init(self)
        for i in range(len(self.frontend.state_dict().items())):
            list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x1 = self.backend_1(x)
        x1 = self.output_layer(x1)
        x2 = self.backend_2(x)
        x2 = self.output_layer(x2)
        x3 = self.backend_3(x)
        x3 = self.output_layer(x3)
        classify = self.srn(x1, x2, x3)
        sig1, sig2, sig3 = self.sig(classify[0])
        x1 = (1 + sig1) * x1
        x2 = (1 + sig2) * x2
        x3 = (1 + sig3) * x3
        x = torch.cat((x1, x2, x3), 1)
        x = self.out_layer(x)
        return x

    def sig(self, classify):
        v1, v2, v3 = classify
        sum = math.exp(v1) + math.exp(v2) + math.exp(v3)
        return v1 / sum, v2 / sum, v3 / sum