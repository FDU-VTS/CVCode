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
warnings.filterwarnings('ignore')
classes = np.asarray(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"])
classes_num = np.asarray([i for i in range(21)])

'''
    PascalVocLoader(image_dir, annotation_path, txt_path, threshold, transform)
        image_dir: JPEGImages path
        annotation_path: Annotations path
        txt_path: aeroplane_train.txt
        threshold: iou threshold
        transform: transform to images
'''


class PascalVocLoader(Dataset):

    def __init__(self, image_dir="./data/VOC2007/JPEGImages/",
                 annotation_path="./data/VOC2007/Annotations/",
                 txt_path="./data/VOC2007/ImageSets/Main/aeroplane_train.txt",
                 threshold=0.5,
                 transform=None):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.txt_path = txt_path
        self.threshold = threshold
        self.transform = transform
        self.dataset = self.xml_reader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        if self.transform:
            data = self.transform(data)
        return data

    # use xml docs to get image:label
    # dataset = [[image, label], [image, label]]
    def xml_reader(self):
        image_dir = self.image_dir
        annotation_path = self.annotation_path
        image_pres = self.txt_reader(self.txt_path)
        ground_truth = []
        # region_set means all images' regions
        region_set = []

        # get every image in image_path
        for image_pre in image_pres:
            image_name = image_pre + ".jpg"
            image_path = image_dir + image_name
            xml_file = annotation_path + image_pre + '.xml'

            # convert xml to document
            dom_tree = xml.dom.minidom.parse(xml_file)
            annotation = dom_tree.documentElement
            object_list = annotation.getElementsByTagName('object')
            image = skimage.io.imread(image_path)

            # selective search
            region_proposal = []
            img, regions = selectivesearch.selective_search(image)
            for region in regions:
                r = region['rect']
                region_proposal.append([[r[0], r[1], r[0] + r[2], r[1] + r[3]],
                                         image[r[1]:r[1]+r[3], r[0]:r[0]+r[2]], 20])

            # each object is a [image: label]
            for object in object_list:
                dom_name = object.getElementsByTagName('name')
                object_name = dom_name[0].childNodes[0].data
                xmin = int(object.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymin = int(object.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmax = int(object.getElementsByTagName('xmax')[0].childNodes[0].data)
                ymax = int(object.getElementsByTagName('ymax')[0].childNodes[0].data)
                image_seg = image[ymin:ymax, xmin:xmax]
                image_label = classes_num[classes == object_name][0]
                ground_truth.append([image_seg, image_label])
                for i in range(len(region_proposal)):
                    iou = utils.get_IoU([xmin, ymin, xmax, ymax], region_proposal[i][0])
                    if iou > self.threshold:
                        region_proposal[i][2] = image_label
            region_proposal = np.asarray(region_proposal)
            if region_set == []:
                region_set = region_proposal
            else:
                region_set = np.append(region_set, region_proposal, axis=0)

            # select only one image to test
            break
        ground_truth = np.asarray(ground_truth)
        return np.append(ground_truth, region_set[:, 1:], axis=0)

    @ staticmethod
    def txt_reader(txt_path):
        txt = np.loadtxt(txt_path, dtype=str)
        image_pres = txt[:, 0]
        return image_pres


# rescale images
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
        image = torch.from_numpy(image)
        return [image, label]