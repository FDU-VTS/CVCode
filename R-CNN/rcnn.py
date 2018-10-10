# -*- coding:utf-8 -*-

import alex_net
import skimage.io
import skimage.transform as transform
import selective_search
import data_loader
import matplotlib.pyplot as plt


if __name__ == "__main__":

    model = alex_net.train()
    image = skimage.io.imread("./lena.jpg")





