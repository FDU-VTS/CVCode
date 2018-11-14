# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------

import os
import skimage.io
from skimage.color import rgb2gray
import skimage.transform
from scipy.io import loadmat
import numpy as np
import cv2
import math
import warnings
import random
warnings.filterwarnings("ignore")


def gaussian_kernel(image, points):
    image_density = np.zeros(image.shape)
    h, w = image_density.shape

    if len(points) == 0:
        return image_density

    for j in range(len(points)):
        f_sz = 15
        sigma = 4.0
        # convert x, y to int
        x = min(w, max(0, int(points[j, 0])))
        y = min(h, max(0, int(points[j, 1])))
        gap = f_sz // 2

        x1 = x - gap if x - gap > 0 else 0
        x2 = x + gap if x + gap < w else w - 1
        y1 = y - gap if y - gap > 0 else 0
        y2 = y + gap if y + gap < h else h - 1
        # generate 2d gaussian kernel
        kx = cv2.getGaussianKernel(y2 - y1 + 1, sigma=sigma)
        ky = cv2.getGaussianKernel(x2 - x1 + 1, sigma=sigma)
        gaussian = np.multiply(kx, ky.T)

        image_density[y1:y2 + 1, x1:x2 + 1] += gaussian

    return image_density


def extract_data(mode="train", patch_number=9, part="A"):
    num_images = 300 if mode=="train" else 182
    # original path
    dataset_path = "../data/original/part_{0}_final/".format(part)
    mode_data = os.path.join(dataset_path, "{0}_data".format(mode))
    mode_images = os.path.join(mode_data, "images")
    mode_ground_truth = os.path.join(mode_data, "ground_truth")
    # preprocessed path
    preprocessed_mode = "../data/preprocessed/{0}/".format(mode)
    preprocessed_mode_density = "../data/preprocessed/{0}_density/".format(mode)
    if not os.path.exists("../data/preprocessed/"):
        os.mkdir("../data/preprocessed/")
    if not os.path.exists(preprocessed_mode):
        os.mkdir(preprocessed_mode)
    if not os.path.exists(preprocessed_mode_density):
        os.mkdir(preprocessed_mode_density)

    # convert images to gray-density for each
    for index in range(1, num_images + 1):
        if index % 10 == 9:
            print("{0} images have been processed".format(index + 1))
        image_path = os.path.join(mode_images, "IMG_{0}.jpg".format(index))
        ground_truth_path = os.path.join(mode_ground_truth, "GT_IMG_{0}.mat".format(index))
        image = skimage.io.imread(image_path)
        # convert to gray map
        if image.shape[-1] == 3:
            image = rgb2gray(image)
        mat = loadmat(ground_truth_path)
        image_info = mat["image_info"]
        ann_points = image_info[0][0][0][0][0]
        # gaussian transfer
        image_density = gaussian_kernel(image, ann_points)
        # split image into 9 patches where patch is 1/4 size
        h, w = image.shape
        w_block = math.floor(w / 8)
        h_block = math.floor(h / 8)
        for j in range(patch_number):
            x = math.floor((w - 2 * w_block) * random.random() + w_block)
            y = math.floor((h - 2 * h_block) * random.random() + h_block)
            image_sample = image[y - h_block:y + h_block, x - w_block:x + w_block]
            image_density_sample = image_density[y - h_block:y + h_block, x - w_block:x + w_block]

            img_idx = "{0}_{1}".format(index, j)
            np.save(os.path.join(preprocessed_mode_density, "{0}.npy".format(img_idx)), image_density_sample)
            skimage.io.imsave(os.path.join(preprocessed_mode, "{0}.jpg".format(img_idx)), image_sample)


def extract_test_data(part="A"):
    num_images = 183 if part == "A" else 317
    test_data_path = "../data/original/part_{part}_final/test_data/images".format(part=part)
    test_ground_path = "../data/original/part_{part}_final/test_data/ground_truth".format(part=part)
    test_density_path = "../data/preprocessed/test_density"
    print("create directory........")
    if not os.path.exists(test_density_path):
        os.mkdir(test_density_path)

    print("begin to preprocess test data........")

    for index in range(1, num_images):
        if index % 10 == 0:
            print("{num} images are done".format(num=index))
        image_path = os.path.join(test_data_path, "IMG_{0}.jpg".format(index))
        ground_truth_path = os.path.join(test_ground_path, "GT_IMG_{0}.mat".format(index))
        # load mat and image
        image = skimage.io.imread(image_path)
        if image.shape[-1] == 3:
            image = rgb2gray(image)
        mat = loadmat(ground_truth_path)
        image_info = mat["image_info"]
        # ann_points: points pixels mean people
        # number: number of people in the image
        ann_points = image_info[0][0][0][0][0]
        number = image_info[0][0][0][0][1]
        h = float(image.shape[0])
        w = float(image.shape[1])
        # convert images to density
        image_density = gaussian_kernel(image, ann_points)
        np.save(os.path.join(test_density_path, "IMG_{0}.npy".format(index)), image_density)


extract_test_data()
