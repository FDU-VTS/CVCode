# -*- coding:utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
import skimage.io
from skimage import transform
import xml.dom.minidom
import warnings
import utils
import selectivesearch
import os
from lxml import etree, objectify
warnings.filterwarnings('ignore')
classes = np.asarray(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"])
classes_num = np.asarray([i for i in range(21)])
IMAGE_DIR = "./data/VOC2007/JPEGImages/"
ANNOTATION_DIR = "./data/VOC2007/Annotations/"
TXT_DIR = "./data/VOC2007/ImageSets/Main/aeroplane_train.txt"

'''
    PascalVocLoader(image_dir, annotation_path, txt_path, threshold, transform)
        image_dir: JPEGImages path
        annotation_path: Annotations path
        txt_path: aeroplane_train.txt
        threshold: iou threshold
        transform: transform to images
'''


def txt_reader(txt_path):
    txt = np.loadtxt(txt_path, dtype=str)
    image_pres = txt[:, 0]
    return image_pres


def load_regions(image_index, number):
    image_name = image_index + ".jpg"
    xml_name = image_index + ".xml"
    image_dir = os.path.join(IMAGE_DIR, image_name)
    xml_dir = os.path.join(ANNOTATION_DIR, xml_name)
    # get regions proposals with selective search
    image = skimage.io.imread(image_dir)
    _, regions = selectivesearch.selective_search(image)
    # get ground truth
    dom_tree = xml.dom.minidom.parse(xml_dir)
    annotation = dom_tree.documentElement
    object_list = annotation.getElementsByTagName('object')
    for object in object_list:
        dom_name = object.getElementsByTagName('name')
        object_name = dom_name[0].childNodes[0].data
        xmin = int(object.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(object.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(object.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(object.getElementsByTagName('ymax')[0].childNodes[0].data)
        ground_truth = [xmin, ymin, xmax, ymax]
        region_proposal = []
        background = []
        for region in regions:
            roi = region['rect']
            iou = utils.get_IoU(ground_truth, roi)
            if iou > 0.5 and len(region_proposal) < number * 0.25:
                label = int(classes_num[classes == object_name])
                region_proposal.append([roi, [label, ground_truth]])
            elif iou > 0.1 and len(background) < number * 0.75:
                background.append([roi, [20, -1]])
            if len(region_proposal) is 16 and len(background) is 48:
                break
    rois = region_proposal + background
    return image, rois


class PascalVocDataset(Dataset):

    def __init__(self, number):
        self.transform = transform
        self.image_indices = txt_reader(TXT_DIR)
        self.number = number

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, item):
        image_index = self.image_indices[item]
        image, rois = load_regions(image_index, self.number)
        image = np.transpose(image, [2, 0, 1])
        image = torch.from_numpy(image)
        return [image, rois]