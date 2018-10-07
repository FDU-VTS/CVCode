# -*- coding:utf-8 -*-
import selectivesearch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# return image regions
def image_proposal(image):
    # selective search
    # img : (width, height, (r, g, b, masked_pixel))
    # regions : list(rect(x_min, y_min, width, height))
    img, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.3 or h / w > 1.3:
            continue
        candidates.add(r['rect'])
    regions = []
    for rect in candidates:
        regions.append(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])

    return regions



