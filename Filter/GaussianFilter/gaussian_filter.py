# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np


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
            sum_x += mask[i]*origin[max(x-i,0)] + mask[i]*origin[min(x+i,length-1)]
        result[x] = sum_x
    return result


def gaussian_filter(origin_image):
    origin_image = np.array(origin_image, dtype=np.float)



print(convolve_1d([1.0, 2.0, 3.0, 4.0, 5.0], 1))