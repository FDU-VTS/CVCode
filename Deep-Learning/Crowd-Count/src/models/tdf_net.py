from __future__ import division
import torch
import torch.nn as nn
from .network import ConvUnit


class BUNet(nn.Module):

    def __init__(self, bn=True):
        super(BUNet, self).__init__()
        self.feature1_1 = nn.Sequential(
            ConvUnit(3, 16, 9, bn=bn),
            nn.MaxPool2d(2)
        )
        self.feature1_2 = ConvUnit(16, 32, 7, bn=bn)
        self.feature1_3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvUnit(32, 16, 7, bn=bn),
        )
        self.final1 = ConvUnit(16, 8, 7, bn=bn)
        self.feature2_1 = nn.Sequential(
            ConvUnit(3, 24, 5, bn=bn),
            nn.MaxPool2d(2)
        )
        self.feature2_2 = ConvUnit(24, 48, 3, bn=bn)
        self.feature2_3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvUnit(48, 24, 3, bn=bn)
        )
        self.final2 = ConvUnit(24, 12, 3, bn=bn)
        self.fuse = nn.Sequential(ConvUnit(20, 1, 1, bn=bn))

    def forward(self, x):
        x1_1 = self.feature1_1(x)
        x1_2 = self.feature1_2(x1_1)
        x1_3 = self.feature1_3(x1_2)
        final1 = self.final1(x1_3)

        x2_1 = self.feature2_1(x)
        x2_2 = self.feature2_2(x2_1)
        x2_3 = self.feature2_3(x2_2)
        final2 = self.final2(x2_3)
        x = torch.cat((x1_3, x2_3), 1)

        out = torch.cat((final1, final2), 1)
        out = self.fuse(out)  # initial density map
        return (out,x,x1_1,x2_1,x1_2,x2_2)


class TDNet(nn.Module):
    
    def __init__(self, bn=True):
        super(TDNet, self).__init__()
        self.BUNet = BUNet()
        self.conv = ConvUnit(40, 16, 3, bn=bn)
        self.pool = nn.MaxPool2d(2,return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        self.output2 = nn.Sequential(
            ConvUnit(96, 32, 3, bn=bn),
            ConvUnit(32, 16, 3, bn=bn),
            nn.Sigmoid()
        )
        self.output3 = nn.Sequential(
            ConvUnit(96, 32, 3, bn=bn),
            ConvUnit(32, 24, 3, bn=bn),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, out_bu, _, _, x1_2, x2_2 = self.BUNet(x)
        out = self.conv(out_bu)
        out, indices = self.pool(out)
        out1 = self.unpool(out, indices)
        unsample = nn.UpsamplingNearest2d(scale_factor=4)
        out1 = unsample(out1)
        x = torch.cat((x1_2, out1, x2_2), 1)
        x1 = self.output2(x)
        x2 = self.output3(x)
        sum_g1 = torch.sum(x1)
        sum_g1 = sum_g1.item()
        sum_g2 = torch.sum(x2)
        sum_g2 = sum_g2.item()
        sum_g = sum_g1 + sum_g2
        return (x1, x2, sum_g)


class TDFNet(nn.Module):

    def __init__(self, bn = True):
        super(TDFNet, self).__init__()
        self.BU = BUNet()
        self.feature1_1 = self.BU.feature1_1
        self.feature2_1 = self.BU.feature2_1
        self.TDNet = TDNet()
        self.reBU_1 = nn.Sequential(
            ConvUnit(16, 32, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(32, 16, 7, bn=bn),
            ConvUnit(16, 8, 7, bn=bn)
        )
        self.reBU_2 = nn.Sequential(
            ConvUnit(24, 48, 3, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(48, 24, 3, bn=bn),  
            ConvUnit(24, 12, 3, bn=bn)
        )
        self.fuse = nn.Sequential(ConvUnit(20, 1, 1, bn=bn))

    def forward(self, x):
        _, bu_out, feature1, feature2, _, _ = self.BU(x)
        x1, x2,_= self.TDNet(x)
        new_x1 = feature1.mul(x1)
        new_x2 = feature2.mul(x2)
        x1 = self.reBU_1(new_x1)
        x2 = self.reBU_2(new_x2)
        x = torch.cat((x1, x2), 1)
        print(x.size())
        x = self.fuse(x)
        return x