# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


# 1 dimension gaussian filter
def convolve_1d(origin, sigma):
    origin = np.array(origin)
    # get the size of mask
    length = np.ceil(sigma*4+1).astype(np.int)
    mask = np.zeros(length)
    sum_norm = 0
    # get gaussian function
    for i in range(length):
        mask[i] = np.exp(-0.5*np.square(i/sigma))
        sum_norm += mask[i]*2
    sum_norm -= mask[0]
    # normalization
    for i in range(length):
        mask[i] /= sum_norm
    # convolve
    result = np.zeros(origin.shape)
    for x in range(len(origin)):
        sum_x = mask[0]*origin[x]
        for i in range(1,length):
            sum_x += mask[i]*origin[max(x-i, 0)] + mask[i]*origin[min(x+i, len(origin)-1)]
        result[x] = sum_x
    return result


def convovle_2d(origin, sigma):
    origin = np.array(origin)
    result = np.zeros(origin.shape)
    for i in range(origin.shape[1]):
        result[:, i] = convolve_1d(origin[:, i], sigma)
    origin = result.T
    result = origin
    for i in range(origin.shape[1]):
        result[:, i] = convolve_1d(origin[:, i], sigma)
    return result.T


def convovle_matrix2d(origin, sigma):
    origin = np.array(origin)
    gaussian = [np.exp(-0.5*np.square(i/sigma)) for i in range(3)]
    gaussian_matrix = np.array([[gaussian[2], gaussian[1], gaussian[2]],
                       [gaussian[1], gaussian[0], gaussian[1]],
                       [gaussian[2], gaussian[1], gaussian[2]]])
    gaussian_matrix = gaussian_matrix / np.sum(gaussian_matrix)
    result = np.zeros(origin.shape)
    width = origin.shape[1]
    height = origin.shape[0]
    for x in range(height):
        for y in range(width):
            result[x, y] = gaussian_matrix[0, 0]*origin[max(x-1, 0), max(y-1, 0)] \
                            + gaussian_matrix[0, 1]*origin[max(x-1, 0), y] \
                            + gaussian_matrix[0, 2]*origin[max(x-1, 0), min(y+1, width-1)] \
                            + gaussian_matrix[1, 0]*origin[x, max(y-1, 0)] \
                            + gaussian_matrix[1, 1]*origin[x, y] \
                            + gaussian_matrix[1, 2]*origin[x, min(y+1, width-1)] \
                            + gaussian_matrix[2, 0]*origin[min(x+1, height-1), max(y-1, 0)] \
                            + gaussian_matrix[2, 1]*origin[min(x+1, height-1), y] \
                            + gaussian_matrix[2, 2]*origin[min(x+1, height-1), min(y+1, width-1)]
    return result

def gaussian_filter():
    a = [[1.0, 2.0, 3.0, 4.0, 5.0],
         [1.0, 2.0, 3.0, 4.0, 5.0],
         [1.0, 2.0, 3.0, 4.0, 5.0],
         [1.0, 2.0, 3.0, 4.0, 5.0],
         [1.0, 2.0, 3.0, 4.0, 5.0]]
    # convolve 1d
    print(convolve_1d(a, 0.8))
    # convolve 2d with 2 1d
    print(convovle_2d(a, 0.8))
    # convolve 2d with mask
    print(convovle_matrix2d(a, 0.8))
    # filter image with 2dmask
    image = np.array(Image.open("./lena.jpg"), dtype=np.float)
    result = convovle_matrix2d(image, 0.8).astype(np.uint8)
    plt.imshow(result)
    plt.show()

gaussian_filter()