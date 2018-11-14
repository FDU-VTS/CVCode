from __future__ import division
import torch
import torch.nn as nn
from src.models.mcnn import ConvUnit as Conv2d


class BUNet(nn.Module):

    def __init__(self):
        super(BUNet, self).__init__()
        self.feature1_1 = nn.Sequential(
            Conv2d(1, 16, 9),
            nn.MaxPool2d(2)
        )
        self.feature1_2 = Conv2d(16, 32, 7)
        self.feature1_3 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv2d(32, 16, 7),
        )
        self.final1 = Conv2d(16, 8, 7)
        self.feature2_1 = nn.Sequential(
            Conv2d(1, 24, 5),
            nn.MaxPool2d(2)
        )
        self.feature2_2 = Conv2d(24, 48, 3)
        self.feature2_3 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv2d(48, 24, 3)
        )
        self.final2 = Conv2d(24, 12, 3)
        self.fuse = nn.Sequential(Conv2d(20, 1, 1))

    def forward(self, x):
        x1_1 = self.feature1_1(x)
        x1_2 = self.feature1_2(x1_1)
        x1_3 = self.feature1_3(x1_2)
        final1 = self.final1(x1_3)

        x2_1 = self.feature2_1(x)
        x2_2 = self.feature2_2(x2_1)
        x2_3 = self.feature2_3(x2_2)
        final2 = self.final2(x2_3)
        x = torch.cat((x1_3, x2_3), 1)  # x, input of top-down

        out = torch.cat((final1, final2), 1)
        out = self.fuse(out)  # initial density map
        return (out, x, x1_1, x2_1, x1_2, x2_2)


class TDNet(nn.Module):
    def __init__(self, bn=True):
        super(TDNet, self).__init__()
        self.BUNet = BUNet()
        self.conv = Conv2d(40, 16, 3)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(4)
        self.output2 = nn.Sequential(
            Conv2d(96, 32, 3),
            Conv2d(32, 16, 3)
        )
        self.output3 = nn.Sequential(
            Conv2d(96, 32, 3),
            Conv2d(32, 24, 3)
        )

    def forward(self, x):
        _, out_bu, _, _, x1_2, x2_2 = self.BUNet(x)
        out = self.conv(out_bu)
        out, indices = self.pool(out)
        out1 = self.unpool(out, indices)
        x = torch.cat((x1_2, out1, x2_2), 1)
        x1 = self.output2(x)
        x2 = self.output3(x)
        return (x1, x2)


class TDFNet(nn.Module):

    def __init__(self):
        super(TDFNet, self).__init__()
        self.BU = BUNet()
        self.feature1_1 = self.BU.feature1_1
        self.feature2_1 = self.BU.feature2_1
        self.TDNet = TDNet()
        self.reBU_1 = nn.Sequential(
            Conv2d(16, 32, 7),
            nn.MaxPool2d(2),
            Conv2d(32, 16, 7),
            Conv2d(16, 8, 7)
        )
        self.reBU_2 = nn.Sequential(
            Conv2d(24, 48, 3),
            nn.MaxPool2d(2),
            Conv2d(48, 24, 3),
            Conv2d(24, 12, 3)
        )
        self.fuse = nn.Sequential(Conv2d(20, 1, 1))

    def forward(self, x):
        _, bu_out, feature1, feature2, _, _ = self.BU(x)
        x1, x2 = self.TDNet(x)
        new_x1 = feature1.mul(x1)
        new_x2 = feature2.mul(x2)
        x1 = self.reBU_1(new_x1)
        x2 = self.reBU_2(new_x2)
        x = torch.cat((x1, x2), 1)
        x = self.fuse(x)
        return x
