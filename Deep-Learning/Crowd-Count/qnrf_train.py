# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils import utils
from src.models import adcrowd_net
from src.datasets import ucf_qnrf
import torch
import torch.utils.data
import torch.optim as optim
import warnings
import sys
import math
import numpy as np
from tensorboardX import SummaryWriter
import os
from torchvision import transforms
import time
import datetime
warnings.filterwarnings("ignore")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(zoom_size=4, model="cbam_net", dataset="ucf_qnrf", learning_rate=1e-7, display=True):
    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    print("train data loading..........")
    train_data = ucf_qnrf.UCFQNRF(mode="train", zoom_size=zoom_size, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    print("test data loading............")
    test_data = ucf_qnrf.UCFQNRF(mode="test", zoom_size=zoom_size, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    # init network
    net = adcrowd_net.ADCrowdNet()
    net = net.train().to(DEVICE)
    # init optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.95, weight_decay=5*1e-4)
    print("start to train net.....")
    time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    model_dir = "{0}_{1}_{2}".format(model, dataset, time_now)
    # whether tensorboardX is needed
    writer = SummaryWriter('runs/' + model_dir) if display else None
    # create model catalog
    if not os.path.exists("./models/{model_name}".format(model_name=model)):
        os.mkdir("./models/{model_name}".format(model_name=model))
    # for each epoch in 2000 get and results to test
    # and keep the best one
    train_loss = utils.LossFunction(model)
    test_loss = utils.LossFunction("test")
    min_mae = sys.maxsize
    min_mse = sys.maxsize
    best_model_name = "{model}_{time}.pkl".format(model=model, time=time_now)
    for epoch in range(2000):
        sum_loss = 0.0
        temp_loss = 0.0
        print("{0} epoches / 2000 epoches are done".format(epoch))
        for i, inputs in enumerate(train_loader):
            for input, ground_truth in inputs:
                input = input.float().to(DEVICE)
                ground_truth = ground_truth.float().to(DEVICE)
                ground_truth = torch.unsqueeze(ground_truth, 0)
                output = net(input)
                loss = train_loss(output, ground_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += float(loss)
            if i % 100 == 99:
                print("{0} / {1} have done, loss is {2}".format(i + 1, len(train_loader), (sum_loss - temp_loss) / 100))
                temp_loss = sum_loss
        # test model
        sum_mae = 0.0
        sum_mse = 0.0
        for inputs in iter(test_loader):
            mae = 0.0
            for input, ground_truth in inputs:
                input = input.float().to(DEVICE)
                ground_truth = ground_truth.float().to(DEVICE)
                output = net(input)
                mae_crop, _ = test_loss(output, ground_truth)
                mae += float(mae_crop)
            mse = mae * mae
            sum_mae += mae
            sum_mse += mse
        if sum_mae / len(test_loader) < min_mae:
            torch.save(net.state_dict(), "./models/{model_name}/{best_model_name}".format(model_name=model, best_model_name=best_model_name))
            min_mae = sum_mae / len(test_loader)
            min_mse = math.sqrt(sum_mse / len(test_loader))
        print("mae:%.1f, mse:%.1f, best_mae:%.1f, best_mse:%.1f" % (sum_mae / len(test_loader),
                                                                    math.sqrt(sum_mse / len(test_loader)), min_mae, min_mse))
        print("sum loss is {0}".format(sum_loss / len(test_loader)))
        if display:
            writer.add_scalar(model_dir + "/loss", np.asarray(sum_loss / len(test_loader), dtype=np.float32), epoch)
            writer.add_scalar(model_dir + "/mae", np.asarray(sum_mae / len(test_loader)), epoch)
            writer.add_scalar(model_dir + "/mse", np.asarray(math.sqrt(sum_mse / len(test_loader))), epoch)
            if epoch == 1999:
                writer.close()


if __name__ == "__main__":
    train()
