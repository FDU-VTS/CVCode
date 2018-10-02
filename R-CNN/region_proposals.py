# -*- coding:utf-8 -*-
import selectivesearch
import skimage
import skimage.transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        if w / h > 3 or h / w > 3:
            continue
        candidates.add(r['rect'])

    return candidates


image = np.array(Image.open("./lena.jpg"))
candidates = image_proposal(image)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(image)
for x, y, w, h in candidates:
    print(x, y, w, h)
    rect = patches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1
    )
    ax.add_patch(rect)

plt.show()

