# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import warnings
import roi_pool
import data_loader
import utils
warnings.filterwarnings('ignore')
classes = np.asarray(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"])
classes_num = np.asarray([i for i in range(21)])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.ROIPool = roi_pool.ROIPool(7, 7, 1/16)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes),
            nn.LogSoftmax()
        )
        self.bbox = nn.Sequential(
            nn.Linear(4096, num_classes * 4)
        )

    def forward(self, x, rois):
        x = self.features(x)
        x = self.ROIPool(x, rois)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        classifier_x = self.classifier(x)
        bbox_x = self.bbox(x)

        return classifier_x, bbox_x


if __name__ == "__main__":
    # load data
    print("start to load data......")
    voc_dataset = data_loader.PascalVocDataset(number=10)
    voc_loader = torch.utils.data.DataLoader(voc_dataset, batch_size=1, shuffle=True, num_workers=0)
    print("init net......")
    net = VGG16(num_classes=21).to(device)
    net = nn.DataParallel(net, device_ids=[0])
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    print("start to train net......")
    for image, image_info in iter(voc_loader):
        rois = [roi[0] for roi in image_info]
        rois_sum = [roi[1] for roi in image_info]
        rois_labels = torch.LongTensor([temp[0] for temp in rois_sum]).to(device)
        ground_truths = [temp[1] for temp in rois_sum]
        image = image.float().to(device)
        rois = torch.FloatTensor(rois).to(device)
        classifier_output, bbox_output = net(image , rois)
        cls_loss = loss_function(classifier_output, rois_labels)
        bbox_loss = utils.bbox_loss(bbox_output, rois, rois_labels, ground_truths)
        loss = cls_loss + bbox_loss
        print("cls_loss: ", cls_loss)
        print("bbox_loss: ", bbox_loss)
        loss.backward()
        optimizer.step()
