# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import os
from dataloader import PascalVOCDataset
import torch
from torch import nn, optim
from torch.utils.data import Dataset
import fast_rcnn_model
from utils import image_list, criterion_loc


def train():

    #init class index
    dataset_index = image_list()
    dataset_index.get_list(os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'Main'))


    #init data
    num_epochs = 10
    train_file_path = os.path.join('VOCdevkit', 'VOC2007', 'Annotations')
    test_file_path = os.path.join('VOCdevkit_test', 'VOC2007', 'Annotations')

    train_img_path = os.path.join('VOCdevkit', 'VOC2007', 'JPEGImages')
    test_img_path = os.path.join('VOCdevkit_test', 'VOC2007', 'JPEGImages')
    voc_datasets = {'train': PascalVOCDataset(train_file_path, dataset_index), 'test': PascalVOCDataset(test_file_path, dataset_index)}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(voc_datasets[x], batch_size=1, shuffle=True, num_workers=4) for x in
        ['train', 'test']}

    #init net
    model_ft = fast_rcnn_model.vgg16(dataset_index, 21, 7, 7, 1.0/16)
    fast_rcnn_model.weights_normal_init(model_ft, dev=0.01)
    fast_rcnn_model.load_pretrained(model_ft)
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft = model_ft.double()
    fast_rcnn_model.set_parameter_requires_grad(model_ft)
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    #model_ft = nn.DataParallel(model_ft, device_ids=(0, 1))

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_loc = nn.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print("strating Itrerate")
            # Iterate over data.
            for inputs, ground_truth, img_path in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                ground_truth = ground_truth.to(device)
                img_path = img_path.to(device)
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':

                        x_cls_prob, x_loc_prob, roi = model_ft(torch.transpose(torch.transpose(inputs, -1, -2), -2, -3).to(device),
                                                                ground_truth, img_path, train_img_path, device)
                        indices = torch.LongTensor([0])
                        labels = torch.index_select(roi, -1, indices)
                        labels = labels.view(-1)
                        labels = torch.tensor(labels, dtype = torch.long)
                        labels = labels.to(device)
                        x_cls_prob = x_cls_prob.view(-1, 21)

                        loss1 = criterion_cls(x_cls_prob, labels)
                        loss2 = criterion_loc(ground_truth, x_loc_prob, roi, labels, device)

                        loss = loss1 + loss2

                    else:
                        x_cls_prob, x_loc_prob, roi = model_ft(
                            torch.transpose(torch.transpose(inputs, -1, -2), -2, -3).to(device), ground_truth, img_path,
                            train_img_path)
                        indices = torch.LongTensor([0])
                        labels = torch.index_select(roi, -1, indices)
                        labels = labels.view(-1)
                        labels = torch.tensor(labels, dtype=torch.long)
                        x_cls_prob = x_cls_prob.view(-1, 21)
                        loss1 = criterion_cls(x_cls_prob, labels)
                        loss = loss1

                    _, preds = torch.max(x_cls_prob, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # # statistics
                print("loss", loss)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(voc_datasets[phase].dataset)
            epoch_acc = running_corrects.double() / len(voc_datasets[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        print()




if __name__ == '__main__':
    train()
