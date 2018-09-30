# -*- coding:utf-8 -*-
import torch
import torch.utils.data
import torch.nn as nn
import torch.functional as f
import torch.multiprocessing
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.cuda
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 5 * 5)
        out = self.dense(x)
        return out


def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # load CIFAR10 dataset
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform)
    # get mini-batch
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False,
                                               num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    time1 = time.time()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(running_loss/2000)
                running_loss = 0.0

    print('cost time: ', time.time()-time1)

