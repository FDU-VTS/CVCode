# -*- coding:utf-8 -*-
import numpy as np


def get_IoU(ground_truth, region):

    # xmin, ymin, xmax, ymax
    x1 = max(ground_truth[0], region[0])
    y1 = max(ground_truth[1], region[1])
    x2 = min(ground_truth[2], region[2])
    y2 = min(ground_truth[3], region[3])

    if x2 - x1 < 0:
        return 0

    inter_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    outer_area = (region[2] - region[0]) * (region[3] - region[1]) \
                 + (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1]) - inter_area
    iou = inter_area / outer_area

    return iou


def NMS(nms_sum):

    regions = []
    nms_sum = nms_sum[nms_sum[:,6]!=20]
    for i in range(len(nms_sum)):
        i_xmin, i_ymin, i_width, i_height, i_image_region, i_score, i_label = nms_sum[i]
        flag = False
        for j in range(len(nms_sum)):
            if i == j:
                continue
            j_xmin, j_ymin, j_width, j_height, j_image_region, j_score, j_label = nms_sum[j]
            iou = get_IoU([i_xmin, i_xmin+i_width, i_ymin, i_ymin+i_height],
                          [j_xmin, j_xmin+j_width, j_ymin, j_ymin+j_height])
            if iou > 0.5 and i_score > j_score:
                flag = True
            elif i_score < j_score:
                break
        if flag == True:
            regions.append([[i_xmin, i_ymin, i_width, i_height], i_label])

    return np.asarray(regions)



