# -*- coding:utf-8 -*-
# ucf_qnrf dataset

from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import cv2
import numpy as np
import h5py
import skimage.io
import skimage.color
import skimage.transform


class UCFQNRF(Dataset):

    def __init__(self, mode="train", **kwargs):
        self.root = "./data/UCF-QNRF_ECCV18/Train/" if mode == "train" else \
                "./data/UCF-QNRF_ECCV18/Test/"
        self.paths = glob.glob(self.root + "*.jpg")
        self.transform = kwargs['transform']
        self.length = len(self.paths)
        self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img, den = self.dataset[item]
        results = []
        h, w, _ = img.shape
        h_crop = h // 4
        w_crop = w // 4
        for i in range(4):
            for j in range(4):
                img_crop = img[h_crop * i: h_crop * (i + 1), w_crop * j: w_crop * (j + 1), :]
                img_crop = cv2.resize(img_crop, (img_crop.shape[1] // 8 * 8, img_crop.shape[0] // 8 * 8), interpolation=cv2.INTER_CUBIC)
                img_crop = self.transform(img_crop)
                den_crop = den[h_crop * i: h_crop * (i + 1), w_crop * j: w_crop * (j + 1)]
                h_trans = den_crop.shape[0] // 8
                w_trans = den_crop.shape[1] // 8
                den_crop = cv2.resize(den_crop, (w_trans, h_trans), interpolation=cv2.INTER_CUBIC) * (den_crop.shape[0] * den_crop.shape[1]) / (h_trans * w_trans)
                results.append([img_crop, den_crop])
        return results

    def load_data(self):
        result = []
        index = 0
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = skimage.io.imread(img_path, plugin='matplotlib')
            img = skimage.color.grey2rgb(img)
            gt_file = h5py.File(gt_path)
            den = np.asarray(gt_file['density'])
            gt_file.close()
            result.append([img, den])
            if index % 100 == 99 or index == self.length:
                print("load {0}/{1} images".format(index + 1, self.length))
            index += 1
        return result


# class UCFQNRF(Dataset):
#
#     def __init__(self, mode="train", **kwargs):
#         self.root = "./data/UCF-QNRF_ECCV18/Train/" if mode == "train" else \
#                 "./data/UCF-QNRF_ECCV18/Test/"
#         self.paths = glob.glob(self.root + "*.jpg")
#         self.transform = kwargs['transform']
#         self.length = len(self.paths)
#         self.dataset = self.load_data()
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, item):
#         img, den = self.dataset[item]
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, den
#
#     def load_data(self):
#         result = []
#         index = 0
#         for img_path in self.paths:
#             gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
#             img = skimage.io.imread(img_path, plugin='matplotlib')
#             img = skimage.color.grey2rgb(img)
#             gt_file = h5py.File(gt_path)
#             den = np.asarray(gt_file['density'])
#             gt_file.close()
#             print(img_path)
#             print(img.shape)
#             img_h, img_w, _ = img.shape
#             zoom_size = 1
#             if img_h < 1000:
#                 pass
#             elif img_h < 3000:
#                 zoom_size = 2
#                 img = cv2.resize(img, (img_w // zoom_size, img_h // zoom_size), cv2.INTER_CUBIC)
#             else:
#                 zoom_size = 4
#                 img = cv2.resize(img, (img_w // zoom_size, img_h // zoom_size), cv2.INTER_CUBIC)
#             h = den.shape[0]
#             w = den.shape[1]
#             h_trans = h // (8 * zoom_size)
#             w_trans = w // (8 * zoom_size)
#             den = cv2.resize(den, (w_trans, h_trans),
#                              interpolation=cv2.INTER_CUBIC) * (h * w) / (h_trans * w_trans)
#             result.append([img, den])
#             if index % 100 == 99 or index == self.length:
#                 print("load {0}/{1} images".format(index + 1, self.length))
#             index += 1
#         return result