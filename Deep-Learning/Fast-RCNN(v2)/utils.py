# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import os
import math
import torch
# selective-search
# return left, top, width, height


def iou(box1, box2):
    top = max(box1[1], box2[1])
    left = max(box1[0], box2[0])
    bottom = min(box1[1]+box1[3], box2[1]+box2[3])
    right = min(box1[0]+box1[2], box2[0]+box2[2])
    if bottom - top < 0 or right - left < 0:
        return 0
    interArea = (bottom - top + 1) * (right - left + 1)

    box1Area = box1[2] * box1[3]
    box2Area = box2[2] * box2[3]
    return float(interArea) / float(box1Area+box2Area-interArea)


class image_list():

    def __init__(self):
        self.img_name = ['background']

    def get_list(self, folder_path):
        file_type = '.txt'
        # Set the groups in a dictionary.
        file_groups = []
        for root, dirs, files in os.walk(folder_path):
            for i in files:
                if file_type in i and '_' in i:
                    name = i.split('_')[0]
                    if name in self.img_name:
                        continue
                    else:
                        self.img_name.append(name)

    def list_all(self):
        print(len(self.img_name))
        for i in range(len(self.img_name)):
            print(self.img_name[i])

    def get_index(self, name):
        return self.img_name.index(name)

    def get_name(self, index):
        return self.img_name[index]


def criterion_loc(ground_truth, x_loc_prob, roi, labels, device):
    loss2 = 0
    batch = x_loc_prob.size()[0]
    num_rois = x_loc_prob.size()[1]
    smoothl1 = torch.nn.SmoothL1Loss(reduction='sum')
    for batch_index in range(batch):
        for roi_index in range(num_rois):
            if labels[batch_index*num_rois+roi_index] == 0:
                continue
            tx = x_loc_prob[batch_index, roi_index, 4 * labels[batch_index*num_rois+roi_index]]
            ty = x_loc_prob[batch_index, roi_index, 4 * labels[batch_index*num_rois+roi_index] + 1]
            tw = x_loc_prob[batch_index, roi_index, 4 * labels[batch_index*num_rois+roi_index] + 2]
            th = x_loc_prob[batch_index, roi_index, 4 * labels[batch_index*num_rois+roi_index] + 3]
            t = torch.tensor((tx, ty, tw, th), dtype=torch.double, device=device, requires_grad=True)
            cls, c, r, w, h = roi[batch_index, roi_index]
            for i in range(ground_truth[batch_index].size()[0]):
                name, xmin, xmax, ymin, ymax = ground_truth[batch_index, i]
                c = int(c)
                r = int(r)
                w = int(w)
                h = int(h)
                ymax = int(ymax)
                ymin = int(ymin)
                xmin = int(xmin)
                xmax = int(xmax)
                if iou((c, r, w, h), (ymin, xmin, xmax - xmin, ymax - ymin)) > 0.4:
                    vx = ((xmax / 2 + xmin / 2) - (c + w / 2)) / w
                    vy = ((ymin / 2 + ymax / 2) - (r + h / 2)) / h
                    vw = math.log((xmax - xmin) / w)
                    vh = math.log((ymax - ymin) / h)
                    v = torch.tensor((vx, vy, vw, vh), dtype=torch.double, device=device)
                    loss2 += smoothl1(t, v)


    return loss2









