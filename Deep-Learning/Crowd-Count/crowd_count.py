# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from src.utils import utils
from src.datasets import mall_dataset, shtu_dataset, shtu_dataset_csr, big_dataset, ucf_cc_50, ucf_qnrf
from src.models import csr_net, sa_net, tdf_net, mcnn, pad_net, vgg, cbam_net, big_net, adcrowd_net
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
models = {
    'mcnn': utils.weights_normal_init(mcnn.MCNN(bn=False), dev=0.01),
    'csr_net': csr_net.CSRNet(),
    'sa_net': sa_net.SANet(input_channels=3, kernel_size=[1, 3, 5, 7], bias=True),
    'tdf_net': utils.weights_normal_init(tdf_net.TDFNet(), dev=0.01),
    'pad_net': pad_net.PaDNet(),
    'vgg': vgg.VGG(),
    'cbam_net': cbam_net.CBAMNet(),
    'big_net': big_net.BIGNet(),
    'adcrowd_net': adcrowd_net.ADCrowdNet()
}


def _load_dataset(dataset, zoom_size=4,transform=None):
    train_loader = None
    test_loader = None
    if dataset == "shtu_dataset":
        print("train data loading..........")
        shanghaitech_dataset = shtu_dataset_csr.ShanghaiTechDataset(mode="train", zoom_size=zoom_size, transform=transform)
        train_loader = torch.utils.data.DataLoader(shanghaitech_dataset, batch_size=1, shuffle=True, num_workers=4)
        print("test data loading............")
        test_data = shtu_dataset_csr.ShanghaiTechDataset(mode="test", transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    elif dataset == "mall_dataset":
        print("train data loading..........")
        mall_data = mall_dataset.MallDataset(img_path="./mall_dataset/frames/", point_path="./mall_dataset/mall_gt.mat", zoom_size=zoom_size)
        train_loader = torch.utils.data.DataLoader(mall_data, batch_size=1, shuffle=True, num_workers=4)
        print("test data loading............")
        mall_test_data = mall_data
        test_loader = torch.utils.data.DataLoader(mall_test_data, batch_size=1, shuffle=False, num_workers=4)
    elif dataset == "big_dataset":
        print("train data loading..........")
        big_data = big_dataset.BigDataset(mode="train", zoom_size=zoom_size, transform=transform)
        train_loader = torch.utils.data.DataLoader(big_data, batch_size=1, shuffle=True, num_workers=4)
        print("test data loading............")
        test_data = big_dataset.BigDataset(mode="test", zoom_size=zoom_size, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    elif dataset == "ucf_cc":
        print("train data loading..........")
        ucf_cc = ucf_cc_50.UCFCC50(mode="train", zoom_size=zoom_size, transform=transform)
        train_loader = torch.utils.data.DataLoader(ucf_cc, batch_size=1, shuffle=True, num_workers=4)
        print("test data loading............")
        test_data = ucf_cc_50.UCFCC50(mode="test", zoom_size=zoom_size, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    elif dataset == "ucf_qnrf":
        print("train data loading..........")
        train_data = ucf_qnrf.UCFQNRF(mode="train", zoom_size=zoom_size, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
        print("test data loading............")
        test_data = ucf_qnrf.UCFQNRF(mode="test", zoom_size=zoom_size, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader


def _init_optimizer(optimizer_name, parameters, learning_rate, momentum=0.95, weight_decay=5*1e-4):
    optimizer = None
    if optimizer_name == "Adam":
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def train(zoom_size=4, model="mcnn", dataset="shtu_dataset", learning_rate=1e-5, optim_name="SGD", pretrain=False, display=True):
    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    train_loader, test_loader = _load_dataset(dataset, zoom_size=zoom_size, transform=transform)
    # init network
    net = models[model]
    if pretrain:
        model_path = "./models/{model}/best_{model}.pkl".format(model=model)
        net.load_state_dict(torch.load(model_path))
    net = net.train().to(DEVICE)
    # init optimizer
    optimizer = _init_optimizer(optim_name, net.parameters(), learning_rate)
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
        for i, (input, ground_truth) in enumerate(train_loader):
            input = input.to(DEVICE)
            ground_truth = ground_truth.float().to(DEVICE)
            ground_truth = torch.unsqueeze(ground_truth, 0)
            output = net(input)
            loss = train_loss(output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
            if i % 100 == 99 or i == len(train_loader) - 1:
                print("{0} / {1} have done, loss is {2}".format(i + 1, len(train_loader), (sum_loss - temp_loss) / 100))
                temp_loss = sum_loss

        # test model
        sum_mae = 0.0
        sum_mse = 0.0
        for input, ground_truth in iter(test_loader):
            input = input.to(DEVICE)
            ground_truth = ground_truth.float().to(DEVICE)
            output = net(input)
            mae, mse = test_loss(output, ground_truth)
            sum_mae += float(mae)
            sum_mse += float(mse)
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
    # args: zoom_size, model, dataset
    print("start....")
    _, model, zoom_size, dataset, learning_rate, optim_name = sys.argv
    print("results: {0}, zoom_size: {1}".format(model, zoom_size))
    train(zoom_size=int(zoom_size), model=str(model), dataset=str(dataset), learning_rate=float(learning_rate), optim_name=str(optim_name))
