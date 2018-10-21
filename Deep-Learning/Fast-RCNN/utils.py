# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn


def get_IoU(ground_truth, region):

    # xmin, ymin, xmax, ymax
    x1 = max(ground_truth[0], region[0])
    y1 = max(ground_truth[1], region[1])
    x2 = min(ground_truth[2], region[0] + region[2])
    y2 = min(ground_truth[3], region[1] + region[3])

    if x2 - x1 < 0:
        return 0

    inter_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    outer_area = (region[2] - region[0] + 1) * (region[3] - region[1] + 1) \
                 + (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] + 1) - inter_area
    if outer_area == 0:
        return 0
    iou = inter_area / outer_area

    return iou


def bbox_loss(bbox_output, rois, roi_labels, ground_truths):
    # output: (20, 4) ground_truth: (, 4)
    bbox_output = bbox_output.view(-1, 4)
    roi_num = rois.size(0)
    loss = 0
    for i in range(roi_num):
        label = roi_labels[i]
        if label == 20:
            continue
        dx, dy, dw, dh = bbox_output[label, :].long()
        Gx, Gy, Gw, Gh = ground_truths[i]
        Px, Py, Pw, Ph = rois[i].long()
        tx = (Gx - Px) / Pw
        ty = (Gy - Py) / Ph
        try:
            tw = math.log(int(Gw) / int(Pw))
            th = math.log(int(Gh) / int(Ph))
        except:
            print("******log exception******")
            print(Gw, Pw, Gh, Ph)
            print(Gw / Pw, Gh / Ph)
            continue
        t = torch.FloatTensor([tx, ty, tw, th])
        d = torch.FloatTensor([dx, dy, dw, dh])
        loss += sum((t - d) ** 2)
    return loss / roi_num


def smooth(x):
    if abs(x) < 1:
        return 0.5 * x ** 2
    else:
        return abs(x) - 0.5
