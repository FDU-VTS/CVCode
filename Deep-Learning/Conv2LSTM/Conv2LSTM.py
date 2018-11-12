# --------------------------------------------------------
# ScaleConvLSTM
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, bias=True):
        super(LSTMCell, self).__init__()

        assert hidden_channels % 2 == 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.Wxi = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Whi = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Wxf = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Whf = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Wxc = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Whc = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Wxo = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Who = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_channels, 480, 640, dtype=torch.double, device=self.device))

    def forward(self, x, h, c):
        print("x", x.size())
        print("wxi", self.Wxi.size())
        print("h", h.size())
        print("c", c.size())
        ci = torch.sigmoid(self.Wxi * x + self.Whi * h + c * self.Wci)
        cf = torch.sigmoid(self.Wxf * x + self.Whf * h + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc * x + self.Whc * h)
        co = torch.sigmoid(self.Wxo * x + self.Who * h + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]).double(),
                torch.zeros(batch_size, hidden, shape[0], shape[1]).double())


class LSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(LSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = LSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


# four scale for conv
class ConvScale(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, bias=True):
        super(ConvScale, self).__init__()
        self.input_channels = input_channels
        self.hidden = max(1, int(input_channels / 2))
        self.output_channels = int(max(1, output_channels / 4))
        self.conv1 = nn.Conv2d(input_channels, self.output_channels, kernel_size[0], 1, int((kernel_size[0] - 1) / 2),
                               bias=bias)
        self.conv2_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2),
                                 bias=bias)
        self.conv3_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2),
                                 bias=bias)
        self.conv4_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2),
                                 bias=bias)
        self.conv2 = nn.Conv2d(self.hidden, self.output_channels, kernel_size[1], 1, int((kernel_size[1] - 1) / 2),
                               bias=bias)
        self.conv3 = nn.Conv2d(self.hidden, self.output_channels, kernel_size[2], 1, int((kernel_size[2] - 1) / 2),
                               bias=bias)
        self.conv4 = nn.Conv2d(self.hidden, self.output_channels, kernel_size[3], 1, int((kernel_size[3] - 1) / 2),
                               bias=bias)
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


class Conv2LSTM(nn.Module):
    def __init__(self, input_channels, kernel_size, bias=True):
        super(Conv2LSTM, self).__init__()
        self.input_channels = input_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.FME = nn.Sequential(
            ConvScale(self.input_channels, 64, self.kernel_size, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            ConvScale(64, 128, self.kernel_size, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            ConvScale(128, 128, self.kernel_size, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            ConvScale(128, 64, self.kernel_size, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            #add
            nn.Conv2d(64, 1, 1, stride=1, padding=int((1 - 1) / 2), bias=True),
            nn.InstanceNorm2d(1, affine=True),
            nn.ReLU(inplace=True),
        )

        self.DME = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=int((9 - 1) / 2), bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 9, stride=2, padding=4, output_padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 7, stride=1, padding=int((7 - 1) / 2), bias=True),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 7, stride=2, padding=3, output_padding=1, bias=True),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 5, stride=1, padding=int((5 - 1) / 2), bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 5, stride=2, padding=2, output_padding=1, bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 3, stride=1, padding=int((3 - 1) / 2), bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 5, stride=1, padding=int((5 - 1) / 2), bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, 1, stride=1, padding=int((1 - 1) / 2), bias=True),
            nn.InstanceNorm2d(1, affine=True),
            nn.ReLU(inplace=True),

        )

        self.LSTM = nn.LSTM(input_size=int(480*640/64), hidden_size=60*80, num_layers=4, batch_first=True, dropout= 0.5)
        # self.LSTM_input_channels = [input_channels] + hidden_channels
        # self.hidden_channels = hidden_channels
        # self.num_layers = len(hidden_channels)
        # self.step = step
        # self.effective_step = effective_step
        # self._all_layers = []
        # for i in range(self.num_layers):
        #     name = 'cell{}'.format(i)
        #     cell = LSTMCell(self.LSTM_input_channels[i], self.hidden_channels[i], self.bias)
        #     setattr(self, name, cell)
        #     self._all_layers.append(cell)

    def forward(self, input):
        # print("in Conv", input.size())
        bsize, frame, channel, height, width = input.size()
        input = input.view(-1, channel, height, width)
        output = self.FME(input)
        # output = self.DME(output)
        # print(output.size())
        output = output.view(bsize, frame, -1)
        output = self.LSTM(output)
        # print(output[0].size())
        # output = output.view(bsize, frame, channel, height, width)
        return output[0].view(bsize, frame, int(height/8), int(width/8))
        # print(output.size())
        # output = output.view(bsize, frame, channel, height, width)
        # print(output.size())
        # internal_state = []
        # outputs = []
        # for step in range(self.step):
        #     x = output[:, step, ..., :]
        #     for i in range(self.num_layers):
        #         # all cells are initialized in the first step
        #         name = 'cell{}'.format(i)
        #         if step == 0:
        #             # bsize, _, height, width = x.size()
        #             (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
        #                                                      shape=(height, width))
        #             internal_state.append((h, c))
        #
        #         # do forward
        #         (h, c) = internal_state[i]
        #         x, new_c = getattr(self, name)(x, h, c)
        #         internal_state[i] = (x, new_c)
        #     # only record effective steps
        #     if step in self.effective_step:
        #         outputs.append(x)
        #
        # return outputs, (x, new_c)


def set_parameter_requires_grad(model, device):
    for name, param in model.named_parameters():
        param.requires_grad = True
        param = param.to(device)
        # print(param.dtype)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # gradient check

    convlstm = Conv2LSTM(input_channels=1, kernel_size=[1, 3, 5, 7]).to(device)
    convlstm = nn.DataParallel(convlstm, device_ids=(0, 1))
    loss_fn = torch.nn.MSELoss()
    convlstm.float()
    input = torch.randn(1, 5, 1, 480, 640, dtype=torch.float).to(device)
    # target = torch.randn((1, 32, 64, 32), dtype  = torch.double)

    output = convlstm(input)
    output = output[0][0].double()
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)
