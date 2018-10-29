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
warnings.filterwarnings("ignore")


def gaussian_kernel(image, points):
    image_density = np.zeros(image.shape)
    h, w = image_density.shape

    if len(points) == 0:
        return image_density

    if len(points) == 1:
        x1 = np.max(0, np.min(w, round(points[0, 0])))
        y1 = np.max(0, np.min(h, round(points[0, 1])))
        image_density[y1, x1] = 255
        return image_density

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
        x1 = x - gap if x - gap > 0 else 0
        x2 = x + gap if x + gap < w else w - 1
        y1 = y - gap if y - gap > 0 else 0
        y2 = y + gap if y + gap < h else h - 1
        # generate 2d gaussian kernel
        kx = cv2.getGaussianKernel(y2 - y1 + 1, sigma=sigma)
        ky = cv2.getGaussianKernel(x2 - x1 + 1, sigma=sigma)
        gaussian = np.multiply(kx, ky.T)

        image_density[y1:y2 + 1, x1:x2 + 1] = image_density[y1:y2 + 1, x1:x2 + 1] + gaussian

    return image_density


def extract_data(mode="train", patch_number=9, part="A"):
    num_images = 300 if mode=="train" else 182
    # original path
    dataset_path = "./data/original/part_{0}_final/".format(part)
    mode_data = os.path.join(dataset_path, "{0}_data".format(mode))
    mode_images = os.path.join(mode_data, "images")
    mode_ground_truth = os.path.join(mode_data, "ground_truth")
    # preprocessed path
    preprocessed_mode = "./data/preprocessed/{0}/".format(mode)
    preprocessed_mode_density = "./data/preprocessed/{0}_density/".format(mode)
    if not os.path.exists("./data/preprocessed/"):
        os.mkdir("./data/preprocessed/")
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
        if image.shape[-1] == 3:
            image = rgb2gray(image)
        mat = loadmat(ground_truth_path)
        image_info = mat["image_info"]
        image_points = image_info[0][0][0][0][0]
        number = image_info[0][0][0][0][1]

        # expand image size to make it can be divided
        h = float(image.shape[0])
        w = float(image.shape[1])
        h_c = int(3 * math.ceil(w / 3))
        w_c = int(3 * math.ceil(h / 3))
        image = skimage.transform.resize(image, (h_c // 4, w_c // 4))
        image_points[:, 0] *= w_c / (4 * w)
        image_points[:, 1] *= h_c / (4 * h)

        image_density = gaussian_kernel(image, image_points)

        image = skimage.transform.resize(image, (h_c, w_c))
        # split an image to 9 patches
        patch_h = int(math.floor(image.shape[0] / 3))
        patch_w = int(math.floor(image.shape[1] / 3))
        points_h = int(math.floor(image_density.shape[0] / 3))
        points_w = int(math.floor(image_density.shape[1] / 3))
        for i in range(3):
            for j in range(3):
                image_partition = image[i * patch_h: (i + 1) * patch_h, j * patch_w: (j + 1) * patch_w]
                density_partition = image_density[i * points_h: (i + 1) * points_h, j * points_w: (j + 1) * points_w]
                skimage.io.imsave("{preprocessed_mode}{index}_{part_num}.jpg".format(preprocessed_mode=preprocessed_mode,
                                                                                     index=index, part_num=i * 3 + j), image_partition)
                np.save("{preprocessed_mode_density}{index}_{part_num}.npy".format(preprocessed_mode_density=preprocessed_mode_density,
                                                                                   index=index, part_num=i * 3 + j), density_partition)


extract_data(mode="test")
