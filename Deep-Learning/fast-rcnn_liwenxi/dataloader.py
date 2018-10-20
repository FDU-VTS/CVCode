# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------

import os
import torch
import xml.etree.ElementTree as ET
from skimage import io, transform
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):

    def __init__(self, xml_path, dataset_index, transform=None):
        self.xml_path = xml_path
        self.dataset_index = dataset_index
        self.object = self.read_xml(xml_path)
        self.transform = transform

    def __len__(self):
        return len(self.object)

    def __getitem__(self, idx):
        image_name = self.object[idx]["image_name"]
        img_path = self.xml_path.replace("/Annotations", "/JPEGImages/")+image_name+".jpg"
        image = io.imread(img_path)
        image = torch.tensor(image, dtype = torch.double)
        ground_truth = self.transport_ground_truth(self.object[idx]["ground_truth"])
        if self.transform:
            image = self.transform(image)
        image_name = torch.tensor((int('1'+image_name), ))
        return image, ground_truth, image_name

    def read_xml(self, xml_path):
        voc_datasets = []

        for root, dirs, files in os.walk(xml_path):
            for i in files:
                voc_datasets.append({"ground_truth": self.get_region(xml_path + '/' + i), "image_name": i.replace(".xml", "")})
        return voc_datasets


    def get_region(self, read_file):
        doc = os.path.abspath(read_file)
        tree = ET.parse(doc)
        root = tree.getroot()
        region = []

        for child in root:
            xmin = 0
            xmax = 0
            ymin = 0
            ymax = 0
            if child.tag == "object":
                for content in child:
                    if content.tag == "name":
                        name = content.text
                    if content.tag == "bndbox":

                        for i in content:
                            if i.tag == "xmin":
                                xmin = int(i.text)
                            if i.tag == "xmax":
                                xmax = int(i.text)
                            if i.tag == "ymin":
                                ymin = int(i.text)
                            if i.tag == "ymax":
                                ymax = int(i.text)
                region.append((name, xmin, xmax, ymin, ymax))
        return region

    def transport_ground_truth(self, ground_truth):
        gt_tensor = []
        for _, ground in enumerate(ground_truth):
            class_index = self.dataset_index.get_index(ground[0])
            ground_truth_elm = (class_index,) + ground[1:]
            gt_tensor.append(ground_truth_elm)
        return torch.tensor(gt_tensor)



def main():
    train_file_path = os.path.join('VOCdevkit', 'VOC2007', 'Annotations')
    datasets = PascalVOCDataset(train_file_path)
    print(datasets.object)

if __name__ == '__main__':
    main()
