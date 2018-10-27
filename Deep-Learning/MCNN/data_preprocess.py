# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

import os
import skimage.io
from scipy.io import loadmat
import numpy as np
import cv2
import math

dataset_path = "./data/part_{0}_final/"
train_data = os.path.join(dataset_path, "train_data")
test_data = os.path.join(dataset_path, "test_data")
train_images = os.path.join(train_data, "images")
train_ground_truth = os.path.join(train_data, "ground_truth")
test_images = os.path.join(test_data, "images")
test_ground_truth = os.path.join(test_data, "ground_truth")


def gaussian_kernel(image, points):
    im_density = np.zeros(image.shape)
    h, w = im_density.shape

    if len(points) == 0:
        return im_density

    if len(points) == 1:
        x1 = np.max(0, np.min(w, round(points[0, 0])))
        y1 = np.max(0, np.min(h, round(points[0, 1])))
        im_density[y1, x1] = 255
        return im_density

    for j in range(len(points)):
        f_sz = 15
        sigma = 4.0
        kx = cv2.getGaussianKernel(f_sz, sigma=sigma)
        ky = cv2.getGaussianKernel(f_sz, sigma=sigma)
        H = np.multiply(kx, ky.T)
        # convert x, y to int
        x = min(w, max(0, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(0, abs(int(math.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        gap = int(math.floor(f_sz / 2))
        x1 = x - gap if x1 >= 0 else 0
        x2 = x + gap if x2 <= w else w
        y1 = y - gap if y1 >= 0 else 0
        y2 = y + gap if y2 <= h else h
        kx = cv2.getGaussianKernel(y2 - y1 + 1, sigma=sigma)
        ky = cv2.getGaussianKernel(x2 - x1 + 1, sigma=sigma)
        gaussian = np.multiply(kx, ky.T)
        im_density[y1:y2, x1:x2] = im_density[y1:y2, x1:x2] + gaussian

    return im_density


for part in ["A", "B"]:
    num_images = 300 if part == "A" else 400
    for index in range(num_images):
        image_path = os.path.join(train_images, "IMG_{0}.jpg".format(index))
        ground_truth_path = os.path.join(train_ground_truth, "GT_IMG_{0}.mat".format(index))
        image = skimage.io.imread(image_path)
        mat = loadmat(ground_truth_path)

        image_annotation = mat[0][0][0][0][0]
        number = mat[0][0][0][0][1]
        


