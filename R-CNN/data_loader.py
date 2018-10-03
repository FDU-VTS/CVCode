# -*- coding:utf-8 -*-
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import skimage.io
import xml.dom.minidom
import matplotlib.pyplot as plt

classes = np.asarray(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
classes_num = np.asarray([i for i in range(20)])


class PascalVocLoader(Dataset):
    def __init__(self, image_dir, annotation_path):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.dataset = self.xml_reader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    # use xml docs to get image:label
    # dataset = [[image, label], [image, label]]
    def xml_reader(self):
        image_dir = self.image_dir
        annotation_path = self.annotation_path
        dataset = []

        # get every image in image_path
        for image_name in os.listdir(image_dir):
            image_path = image_dir + image_name
            image_pre, _ = os.path.splitext(image_name)
            xml_file = annotation_path + image_pre + '.xml'

            # convert xml to document
            dom_tree = xml.dom.minidom.parse(xml_file)
            annotation = dom_tree.documentElement
            object_list = annotation.getElementsByTagName('object')
            image = skimage.io.imread(image_path)

            # each object is a [image: label]
            for object in object_list:
                dom_name = object.getElementsByTagName('name')
                object_name = dom_name[0].childNodes[0].data
                xmin = int(object.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymin = int(object.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmax = int(object.getElementsByTagName('xmax')[0].childNodes[0].data)
                ymax = int(object.getElementsByTagName('ymax')[0].childNodes[0].data)
                image_seg = image[xmin:xmax, ymin:ymax]
                image_label = classes_num[classes == object_name]
                dataset.append((image_seg, image_label))

        return dataset


dataset = PascalVocLoader(image_dir="./data/VOC2007/JPEGImages/", annotation_path="./data/VOC2007/Annotations/")
