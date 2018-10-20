# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from roi_pooling import RoI_Pooling
import selectivesearch
from skimage import io
import random
import utils


class vgg16(nn.Module):

    def __init__(self, dataset_index, num_classes=1000, pooled_height=7, pooled_width=7, spatial_scale=1.0/16):
        super(vgg16, self).__init__()
        self.dataset_index = dataset_index
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)


        )
        self.roi_pooling = RoI_Pooling(pooled_height, pooled_width, spatial_scale)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.cls = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.loc = nn.Linear(in_features=4096, out_features=num_classes*4, bias=True)

    def forward(self, x, ground_truth, img_name, img_path, device):

        roi = torch.zeros((img_name.size()[0],)+(64,5))
        for img_indx, this_img_name in enumerate(img_name):
            name = img_path + '/' + str(int(img_name[img_indx]))[1:] + '.jpg'
            for i in enumerate(img_name):
                roi_elem = self.ss(name, ground_truth)
            roi[img_indx, :, :] = roi_elem
        x = self.features(x)
        x = self.roi_pooling(x, roi)
        x = x.view(tuple(x.size()[:-3]) + (-1, ))
        x = x.to(device)
        x = self.classifier(x)
        x_cls_score = self.cls(x)
        x_cls_prob = F.softmax(x_cls_score, dim=-1)
        x_loc_prob = self.loc(x)
        return x_cls_prob, x_loc_prob, roi

    def ss(self, img_path, ground_truth):
        img = io.imread(img_path)
        img_lbl, regions = selectivesearch.selective_search(
            img)
        candidates = set()
        roi_positive = []
        roi_negative = []
        roi_backup = []

        for r in regions:
            # excluding same rectangle (withsl different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 500:
                continue
            candidates.add(r['rect'])
            for i, ground in enumerate(ground_truth):
                class_index, xmin, xmax, ymin, ymax = ground[0]
                if utils.iou(r['rect'], (xmin.item(), ymin.item(), xmax.item()-xmin.item(),ymax.item()-ymin.item())) > 0.4:
                    roi_positive.append((class_index,) + r['rect'])
                elif utils.iou(r['rect'], (ymin.item(), xmin.item(), xmax.item()-xmin.item(),ymax.item()-ymin.item())) > 0.1:
                    roi_negative.append((0,) + r['rect'])
                else:
                    roi_backup.append((0,) + r['rect'])
    
        positive_num = min(len(roi_positive), 16)
        negative_num = min(len(roi_negative), 64-positive_num)
        roi = random.sample(roi_positive, positive_num)
        roi += random.sample(roi_negative, negative_num)
        if positive_num+negative_num < 64:
            roi += random.sample(roi_backup, 64-positive_num-negative_num)

        roi = torch.tensor(roi)
        return roi

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def load_pretrained(model_ft):
    model_dict = model_ft.state_dict()

    pretrained_dict = models.vgg16(pretrained=True).state_dict()
    # print(models.vgg16(pretrained=True))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model_ft.load_state_dict(model_dict)

def set_parameter_requires_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
        # param = param.type('torch.DoubleTensor')
    feature = list(model.features)
    for layer in feature[:6]:
        for p in layer.parameters():
            p.requires_grad = False

