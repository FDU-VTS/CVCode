# -*- coding:utf-8 -*-

import torch.nn as nn

'''
Conv/Pooling/any other layer: 
    (input_size + 2*padding - kernel_size) / stride = output_size
    
Alex Net architecture:
    Convolutional layer 1:
        input: 224 * 224 * 3
        Conv1: (96 kernels | 11 * 11 * 3 kernel size | 4 strides | padding = 2)
        output: 55 * 55 * 48 * 2 (2 GPU for 2 same nets)
    Pooling and Normalization:
        ReLu: 55 * 55 * 48 * 2 -> 55 * 55 * 48 * 2
        MaxPooling: 55 * 55 * 48 * 2 -> (3 * 3 | 2 strides) -> 27 * 27 * 48 * 2
    Convolutional layer 2:
        input: 27 * 27 * 48 * 2
        Conv2: (256 kernels | 5 * 5 * 48 kernel size | 1 strides | padding = 2)
        output: 27 * 27 * 128 * 2
    Pooling and Normalization:
        ReLu: 27 * 27 * 128 * 2 -> 27 * 27 * 128 * 2
        MaxPooling: 27 * 27 * 128 * 2 -> (3 * 3 | 2 strides) -> 13 * 13 * 128 * 2
    Convolutional layer 3:
        input: 13 * 13 * 128 * 2
        Conv3: (384 kernels | 3 * 3 * 128 kernel size | 1 strides | padding = 1)
        output: 13 * 13 * 192 * 2
    Pooling and Normalization:
        ReLu: 13 * 13 * 192 * 2 -> 13 * 13 * 192 * 2
    Convolutional layer 4:
        input: 13 * 13 * 192 * 2
        Conv4: (384 kernels | 3 * 3 * 192 kernel size | 1 strides | padding = 1)
        output: 13 * 13 * 192 * 2
    Pooling and Normalization:
        ReLu: 13 * 13 * 192 * 2 -> 13 * 13 * 192 * 2
    Convolutional layer 5:
        input: 13 * 13 * 192 * 2
        Conv5: (256 kernels | 3 * 3 * 192 kernel size | 1 strides | padding = 1)
        output: 13 * 13 * 128 * 2
    Pooling and Normalization:
        ReLu: 13 * 13 * 128 * 2 -> 13 * 13 * 128 * 2
        MaxPooling: 13 * 13 * 128 * 2 -> (3 * 3 | 2 strides) -> 6 * 6 * 128 * 2
    Full connection layer 1:
        input: 6 * 6 * 128 * 2
        output: 4096
    Full connection layer 2:
        input: 4096
        output: 4096
    Full connection layer 3:
        input: 4096
        output: 1000 
'''


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



