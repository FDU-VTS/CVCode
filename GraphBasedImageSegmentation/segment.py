# -*- coding:utf-8 -*-

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from segment_image import *


def image_segment(image_name, sigma, k, min_size, num_css):
    image_origin = np.array(Image.open(image_name))
    seg = segment_image(image_origin,sigma,k,min_size,num_css)
    seg = np.array(seg, dtype=np.int)
    plt.figure("Image")
    plt.imshow(seg)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_name = "./lena.jpg"
    sigma = 0.8
    k = 500
    min_size = 50
    num_css = 100
    image_segment(image_name, sigma, k, min_size, num_css)