import torch
import torch.nn as nn
# from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = torch.zeros(1, hidden, shape[0], shape[1])
        self.Wcf = torch.zeros(1, hidden, shape[0], shape[1])
        self.Wco = torch.zeros(1, hidden, shape[0], shape[1])
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]),
                torch.zeros(batch_size, hidden, shape[0], shape[1]))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
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
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
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
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
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
        self.hidden = max(1, int(input_channels/2))
        self.conv1 = nn.Conv2d(input_channels, int(output_channels/4), kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv2_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv3_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv4_1 = nn.Conv2d(input_channels, self.hidden, kernel_size[0], 1, int((kernel_size[0] - 1) / 2), bias=bias)
        self.conv2 = nn.Conv2d(self.hidden, int(output_channels/4), kernel_size[1], 1, int((kernel_size[1] - 1) / 2), bias=bias)
        self.conv3 = nn.Conv2d(self.hidden, int(output_channels/4), kernel_size[2], 1, int((kernel_size[2] - 1) / 2), bias=bias)
        self.conv4 = nn.Conv2d(self.hidden, int(output_channels/4), kernel_size[3], 1, int((kernel_size[3] - 1) / 2), bias=bias)

    def forward(self, input):
        output1 = self.conv1(input)
        output2 = self.conv2_1(input)
        output3 = self.conv3_1(input)
        output4 = self.conv4_1(input)

        output2 = self.conv2(output2)
        output3 = self.conv3(output3)
        output4 = self.conv4(output4)
        # print(output1.size())
        # print(output2.size())
        # print(output3.size())
        # print(output4.size())
        return torch.cat([output1, output2, output3, output4], -3)

class ScaleConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ScaleConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4


        self.Wxi = ConvScale(self.input_channels, self.hidden_channels, self.kernel_size, bias=True)
        self.Whi = ConvScale(self.hidden_channels, self.hidden_channels, self.kernel_size, bias=False)
        self.Wxf = ConvScale(self.input_channels, self.hidden_channels, self.kernel_size, bias=True)
        self.Whf = ConvScale(self.hidden_channels, self.hidden_channels, self.kernel_size, bias=False)
        self.Wxc = ConvScale(self.input_channels, self.hidden_channels, self.kernel_size, bias=True)
        self.Whc = ConvScale(self.hidden_channels, self.hidden_channels, self.kernel_size, bias=False)
        self.Wxo = ConvScale(self.input_channels, self.hidden_channels, self.kernel_size, bias=True)
        self.Who = ConvScale(self.hidden_channels, self.hidden_channels, self.kernel_size, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = torch.zeros(1, hidden, shape[0], shape[1])
        self.Wcf = torch.zeros(1, hidden, shape[0], shape[1])
        self.Wco = torch.zeros(1, hidden, shape[0], shape[1])
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]),
                torch.zeros(batch_size, hidden, shape[0], shape[1]))


class ScaleConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ScaleConvLSTM, self).__init__()
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
            cell = ScaleConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
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
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


if __name__ == '__main__':

    # gradient check
    convlstm = ScaleConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=[1, 3, 5, 7], step=5, effective_step=[4])
    loss_fn = torch.nn.MSELoss()

    input = torch.randn(1, 512, 64, 32)
    target = torch.randn((1, 32, 64, 32), dtype  = torch.double)

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)