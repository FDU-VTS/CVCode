# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from net.roi_pool import roi_pooling
import skimage.io
import numpy as np
import data_loader
import torchvision.transforms as transforms
import selectivesearch
import matplotlib.pyplot as plt
import warnings
import utils
warnings.filterwarnings('ignore')
classes = np.asarray(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"])
classes_num = np.asarray([i for i in range(21)])


class VGG16(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ROIPool = roi_pooling.ROIPooling(7, 7, 1/16)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(),
        )

    def forward(self, x, rois = None, ):
        x = self.features(x)
        print("conv layer ouput: ", x.size())
        x = self.ROIPool(x, rois)
        print("ROI pool output: ", x.size())
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        print("Linear layer output: ", x.size())

        return x



if __name__ == "__main__":
    # load data
    image = skimage.io.imread("../data/car.jpg")
    # ground truth
    ground_truth = [156, 97, 351, 270]
    ground_truth_label = "car"
    ground_truth_number = classes_num[classes=="car"]
    # get regions
    region_proposal = []
    rois_labels = []
    background = []
    img, regions = selectivesearch.selective_search(image)
    for region in regions:
        rect = region['rect']
        roi = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        if roi in region_proposal or roi in background:
            continue
        iou = utils.get_IoU(ground_truth, roi)
        if iou > 0.5 and len(region_proposal) < 16:
            region_proposal.append(roi)
            rois_labels.append(ground_truth_number)
        elif iou > 0.1 and len(background) < 48:
            background.append(roi)
        if len(region_proposal) is 16 and len(background) is 48:
            break

    net = VGG16(num_classes=21)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # transfer data to tensor
    image = image.transpose([2, 0, 1])
    image = torch.from_numpy(image)
    image = image.view([1] + list(image.size())).float()
    rois = region_proposal + background
    rois_labels += [20 for i in range(len(background))]
    rois = torch.Tensor(rois)
    print("rois' size: ", rois.size())
    output = net(image, rois)
    print("output size: ", output.size())
    print("backward......")
    for output_labels, labels in zip(output, rois_labels):
        optimizer.zero_grad()
        labels = torch.from_numpy(labels)
        output_labels = output_labels.view(1, -1)
        print(output_labels.size(), labels.size())
        loss = loss_function(output_labels, labels)
        loss.backward()
        optimizer.step()
        print("loss: ", loss.item())
        break
