# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import warnings
import roi_pool
import data_loader
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
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(),
        )
        self.bbox = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, (num_classes - 1) * 4)
        )

    def forward(self, x, rois=None):
        x = self.features(x)
        print("conv layer ouput: ", x.size())
        print("ROI pool start...")
        x = self.ROIPool(x, rois)
        print("ROI pool output: ", x.size())
        x = x.view(x.size(0), -1)
        classifier_x = self.classifier(x)
        bbox_x = self.bbox(x)

        return classifier_x, bbox_x


if __name__ == "__main__":
    # load data
    voc_dataset = data_loader.PascalVocDataset(number=64)
    voc_loader = torch.utils.data.DataLoader(voc_dataset, batch_size=1, shuffle=True, num_workers=2)
    net = VGG16(num_classes=21).to(device)
    net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for image, image_info in iter(voc_loader):
        rois = [roi[0] for roi in image_info]
        rois_labels = [roi[1] for roi in image_info]
        rois_label = torch.LongTensor([label[0] for label in rois_labels])
        ground_truth = [label[1] for label in rois_labels]
        image = image.float().to(device)
        rois = torch.FloatTensor(rois).to(device)
        classifier_output, bbox_output = net(image , rois)
        cls_loss = loss_function(classifier_output, rois_label)
        cls_loss.backward()
        print("cls_loss: ", cls_loss)
        optimizer.step()
