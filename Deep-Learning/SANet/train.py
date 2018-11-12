# --------------------------------------------------------
# SANet
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import torch
from torch import nn, optim
from torchvision import models
from DataLoader import MallDataset, MallDatasetTest
from SANet import SANet, set_parameter_requires_grad
import math
import copy
import time
# from tensorboardX import SummaryWriter

def main():
    # writer = SummaryWriter('tensorboard.log')
    num_epochs = 1000
    img_path = "./mall_dataset/frames/"
    point_path = './mall_dataset/mall_gt.mat'
    dataset = MallDataset(img_path, point_path)
    dataset_test = MallDatasetTest(img_path, point_path)
    dataloader= torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=16)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=5, shuffle=False, num_workers=16)
    # model = ScaleConvLSTM(input_channels=1, hidden_channels=[128, 64, 64, 32, 32], kernel_size=[1, 3, 5, 7], step=5, effective_step=[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SANet(input_channels=1, kernel_size=[1, 3, 5, 7], bias=True).to(device)

    model = model.double()
    model.load_state_dict(torch.load('INMAE12325.399114746237MSE441018.38155946246best_model_wts.pkl'))
    model = nn.DataParallel(model, device_ids=(0, 1))

    #set_parameter_requires_grad(model, device)
    # print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # print("\t", name)
            # print("\t", param.dtype)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    MSEloss = nn.MSELoss()
    L1loss = nn.L1Loss()
    MAE_best = 12463.0000
    best_model_wts = copy.deepcopy(model.state_dict())
    lossoutput = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start_time = time.time()
        # for phase in ['test', 'train']:

        for phase in ['train','test']:
            print("strating Itrerate")
            running_loss = 0.0
            train_start_time = time.time()
            MAE = 0
            MSE = 0
            if phase == 'train':
                model.train()  # Set model to training mode
                step = 0;
                for inputs, ground_truth in dataloader:
                    step_time = time.time()
                    print("Epoch {} Train Step {}: ".format(epoch, step))
                    step+=1
                    # if step % 20 == 0:
                    #     backup_model_wts = copy.deepcopy(model.state_dict())
                    #     torch.save(backup_model_wts, str(lossoutput)+'backup_model_wts.pkl')
                    # inputs = inputs.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            inputs = inputs.to(device)
                            ground_truth = ground_truth.to(device)
                            outputs = model(inputs)
                            loss = MSEloss(outputs, ground_truth)
                            print("Loss: ", loss.item())
                            lossoutput = loss.item()

                        # backward + optimize only if in training phase

                            print("begin backword")
                            loss.backward()
                            print("begin optimizer")
                            optimizer.step()

                    # # statistics
                    running_loss += loss.item() * inputs.size(0)
                    print("This Step Used", time.time() - step_time)
                print("This Train Used", time.time() - train_start_time)
            else:
                print("starting test")
                model.eval()  # Set model to evaluate mode
                torch.set_grad_enabled(False)
                step = 0;
                for inputs, ground_truth in dataloader_test:
                    step_time = time.time()
                    print("Epoch {} Test Step {}: ".format(epoch, step))
                    step+=1

                    inputs = inputs.to(device)
                    ground_truth = ground_truth.to(device)

                    outputs = model(inputs)
                    outputs = torch.sum(outputs, (-1, -2))
                    MAE += L1loss(outputs, ground_truth)
                    MSE += MSEloss(outputs, ground_truth)
                    print("MAE", MAE.item()/step)
                    print("MSE", MSE.item()/step)
                    print("This Step Used", time.time() - step_time)
                print("MAE", MAE.item()/2000)
                print("MSE", math.sqrt(MSE.item()/2000))
                print("This Test Used", time.time() - train_start_time)
                if MAE < MAE_best:
                    MAE_best = MAE
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, "IN"+"MAE"+str(MAE.item())+"MSE"+str(MSE.item())+'best_model_wts.pkl')



            # statistics
            # running_loss += loss.item() * inputs.size(0)

            # epoch_loss = running_loss / len(dataset.__len__())

            # print('{} Loss: {:.4f} '.format(phase, epoch_loss))

        print("This Epoch used", time.time()-start_time)
        print()


if __name__ == '__main__':
    main()
