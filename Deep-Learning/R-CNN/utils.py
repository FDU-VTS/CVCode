# -*- coding:utf-8 -*-


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

def NMS():
    pass