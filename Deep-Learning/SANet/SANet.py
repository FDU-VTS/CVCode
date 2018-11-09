# --------------------------------------------------------
# SANet
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import torch
from torch import nn

# four scale for conv
class ConvScale(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, bias=True):
        super(ConvScale, self).__init__()
        self.input_channels = input_channels
        self.hidden = max(1, int(input_channels/2))
        self.output_channels = int(max(1, output_channels/4))
        self.conv1 = nn.Conv2d(input_channels, self.output_channels, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv2_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv3_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv4_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv2 = nn.Conv2d(self.hidden, self.output_channels, kernel_size[1], 1, int((kernel_size[1] - 1) / 2), bias=bias)
        self.conv3 = nn.Conv2d(self.hidden, self.output_channels, kernel_size[2], 1, int((kernel_size[2] - 1) / 2), bias=bias)
        self.conv4 = nn.Conv2d(self.hidden, self.output_channels, kernel_size[3], 1, int((kernel_size[3] - 1) / 2), bias=bias)
        self.conv_result = nn.Conv2d(4, 1, 1, stride=1, padding=0, bias=bias)

    def forward(self, input):
        # input.double()finish ci
        # print("forward size", input.dtype)
        # print("start ConvScale")
        output1 = self.conv1(input)
        if self.input_channels == 1:
            output2 = self.conv2(input)
            output3 = self.conv3(input)
            output4 = self.conv4(input)
            if self.output_channels == 1:
                return nn.ReLU()(self.conv_result(torch.cat([output1, output2, output3, output4], -3)))
            return nn.ReLU()(torch.cat([output1, output2, output3, output4], -3))

        # print("finish step 1")
        output2 = self.conv2_1(input)
        output3 = self.conv3_1(input)
        output4 = self.conv4_1(input)
        # print("finish step 2")
        output2 = self.conv2(output2)
        output3 = self.conv3(output3)
        output4 = self.conv4(output4)
        # print("finish ConvScale")
        if self.output_channels == 1:
            return nn.ReLU()(self.conv_result(torch.cat([output1, output2, output3, output4], -3)))
        return nn.ReLU()(torch.cat([output1, output2, output3, output4], -3))


class SANet(nn.Module):
    def __init__(self, input_channels, kernel_size, bias=True):
        super(SANet, self).__init__()

        self.input_channels = input_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.FME = nn.Sequential(
            ConvScale(self.input_channels, 64, self.kernel_size, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            ConvScale(64, 128, self.kernel_size, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            ConvScale(128, 128, self.kernel_size, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            ConvScale(128, 64, self.kernel_size, bias=True),
            nn.ReLU(inplace=True),
        )

        self.DME = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=int((9 - 1) / 2), bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 9, stride=2, padding=4, output_padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 7, stride=1, padding=int((7 - 1) / 2), bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 7, stride=2, padding=3, output_padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 5, stride=1, padding=int((5 - 1) / 2), bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 5, stride=2, padding=2, output_padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 3, stride=1, padding=int((3 - 1) / 2), bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 5, stride=1, padding=int((5 - 1) / 2), bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, 1, stride=1, padding=int((1 - 1) / 2), bias=True),
        )

    def forward(self, input):
        output = self.FME(input)
        output = self.DME(output)
        return output

def set_parameter_requires_grad(model, device):
    for name, param in model.named_parameters():
        param.requires_grad = True
        param = param.to(device)
        # print(param.dtype)
