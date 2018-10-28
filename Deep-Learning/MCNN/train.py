# -*- coding:utf-8 -*-
import mcnn
import shtu_dataset
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import utils
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    print("data loading..........")
    shanghaitech_dataset = shtu_dataset.ShanghaiTechDataset()
    tech_loader = torch.utils.data.DataLoader(shanghaitech_dataset, batch_size=3, num_workers=2)
    print("init net...........")
    net = mcnn.MCNN().to(DEVICE)
    if torch.cuda.is_available():
        net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    print("start to train net.....")
    sum_loss = 0
    i = 0
    for epoch in range(20):
        for input, ground_truth in iter(tech_loader):

            input = input.float().to(DEVICE)
            ground_truth = ground_truth.float().to(DEVICE)
            output = net(input)
            loss = utils.get_loss(output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss
            i += 1
            if i % 50 == 49:
                print("loss: ", sum_loss / 50)
                sum_loss = 0

    return net

if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "./model/mcnn.pkl")