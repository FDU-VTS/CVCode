import torch.nn as nn
from torchvision import models
from src.models.network import ConvUnit
import torch
from src.utils import utils
import torch.nn.functional as F
from collections import OrderedDict
import torchvision


class ADCrowdNet(nn.Module):
    def __init__(self, bn=True,load_weights=False):
        super(ADCrowdNet, self).__init__()
        # self.frontend= [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.frontend_feature = make_layers(self.frontend)
        self.mydensenet = torchvision.models.densenet121(pretrained=False)
        self.densenet_layer = nn.Sequential(*list(self.mydensenet.children())[:-1][0][:6])

        self.back_Dconv1_1 = ConvUnit(128, 256, 3, stride=1)
        self.back_Dconv1_2 = ConvUnit(128, 256, 5, stride=1)
        self.back_Dconv1_3 = ConvUnit(128, 256, 7, stride=1)

        self.back_Dconv2_1 = ConvUnit(256, 128, 3, stride=1)
        self.back_Dconv2_2 = ConvUnit(256, 128, 5, stride=1)
        self.back_Dconv2_3 = ConvUnit(256, 128, 7, stride=1)

        self.back_Dconv3_1 = ConvUnit(128, 64, 3, stride=1)
        self.back_Dconv3_2 = ConvUnit(128, 64, 5, stride=1)
        self.back_Dconv3_3 = ConvUnit(128, 64, 7, stride=1)


        self.fuse1 =ConvUnit(768, 256, 1, bn=bn)
        self.fuse2 = ConvUnit(384, 128, 1, bn=bn)
        self.fuse3 = ConvUnit(192, 1, 1, bn=bn)

        # if not load_weights:
        #     mod = models.densenet_layer(pretrained=False)
        #     utils.weights_normal_init(self)
        #     for i in range(len(self.features.state_dict().items())):
        #         list(self.features.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        # x = self.frontend_feature(x)
        x = self.densenet_layer(x)
        x1_1 = self.back_Dconv1_1(x)
        x1_2 = self.back_Dconv1_2(x)
        x1_3 = self.back_Dconv1_3(x)
        fc1 = torch.cat((x1_1, x1_2, x1_3), 1)
        out1 = self.fuse1(fc1)  # torch.Size([1, 256, 20, 20])

        x2_1 = self.back_Dconv2_1(out1)
        x2_2 = self.back_Dconv2_2(out1)
        x2_3 = self.back_Dconv2_3(out1)
        fc2 = torch.cat((x2_1, x2_2, x2_3), 1)
        out2 = self.fuse2(fc2)  # torch.Size([1, 128, 10, 10])

        x3_1 = self.back_Dconv3_1(out2)
        x3_2 = self.back_Dconv3_2(out2)
        x3_3 = self.back_Dconv3_3(out2)
        fc3 = torch.cat((x3_1, x3_2, x3_3), 1)
        out3 = self.fuse3(fc3)  # torch.Size([1, 1, 5, 5])
        return out3

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