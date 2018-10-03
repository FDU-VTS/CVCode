# -*- coding:utf-8 -*-
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import skimage.io
from skimage import transform
import xml.dom.minidom
import matplotlib.pyplot as plt

classes = np.asarray(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
classes_num = np.asarray([i for i in range(20)])


class PascalVocLoader(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.transform = transform
        self.dataset = self.xml_reader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = dataset[item]
        data = self.transform(data)
        return data

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
            break

        return dataset


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return [img, label]


class ToTensor(object):

    def __call__(self, sample):
        image, label = sample
        image = image.transpose([2, 0, 1])
        return [image, label]

dataset = PascalVocLoader(image_dir="./data/VOC2007/JPEGImages/",
                          annotation_path="./data/VOC2007/Annotations/")
composed = transforms.Compose([Rescale((224, 224)), ToTensor()])
a = composed(dataset[0])
