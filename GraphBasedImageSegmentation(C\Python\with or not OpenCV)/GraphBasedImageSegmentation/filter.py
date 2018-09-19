# -*- coding:utf-8 -*-

import numpy as np
from convolve import *

WIDTH = 4


class Filter:

    @staticmethod
    def normalize(mask):
        length = len(mask)
        sum = 0
        for i in range(1,length):
            sum += abs(mask[i])
        sum = 2*sum + abs(mask[0])
        for i in range(length):
            mask[i] /= sum
        return mask

    @staticmethod
    def laplacian(image_origin):

        image_lap = image_origin
        image_shape = image_origin.shape
        width = image_shape[0]
        height = image_shape[1]

        for y in range(height[1,-1]):
            for x in range(width[1,-1]):
                image_lap[x,y] = image_origin[x-1,y] + image_origin[x+1,y] + image_origin[x,y-1] \
                                 + image_origin[x,y+1]- 4*image_origin[x,y]

        return image_lap

    @staticmethod
    def make_gauss(sigma):
        sigma = max(sigma,0.01)
        length = int(np.ceil(sigma * WIDTH) + 1)
        mask = np.zeros(length)
        for i in range(length):
            mask[i] = np.exp(-0.5 * np.square(i/sigma))
        return mask

    @staticmethod
    def smooth(src,sigma):
        mask = Filter.make_gauss(sigma)
        mask = Filter.normalize(mask)
        tmp = np.zeros(src.T.shape)
        dst = np.zeros(src.shape)
        tmp = convolve_even(src,tmp,mask)
        dst = convolve_even(tmp,dst,mask)
        return dst



