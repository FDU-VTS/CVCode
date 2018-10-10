# -*- coding:utf-8 -*-


def convolve_even(src,dst,mask):

    for y in range(src.shape[1]):
        for x in range(src.shape[0]):
            sum = mask[0] * src[x,y]
            for i in range(1,len(mask)):
                sum += mask[i] * (int(src[max(x - i, 0), y]) + int(src[min(x + i, src.shape[0] - 1), y]))
            dst[y, x] = sum

    return dst


def convolve_odd(src,dst,mask):

    for y in range(src.shape[1]):
        for x in range(src.shape[0]):
            sum = mask[0] * src[x, y]
            for i in range(1,src.shape[0]):
                sum += mask[i] * (int(src[max(x - i, 0), y]) + int(src[min(x + i, src.shape[0] - 1), y]))
            dst[y, x] = sum

    return dst