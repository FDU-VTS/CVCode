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



