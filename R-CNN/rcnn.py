# -*- coding:utf-8 -*-

import alex_net
import skimage.io
import skimage.transform as transform
import selective_search


if __name__ == "__main__":

    model = alex_net.train()
    image = skimage.io.imread("./lena.jpg")
    # regions: image[x1:x2, y1:y2]
    regions = selective_search.region_proposals(image)
    for region in regions:
        # region: [x_min, y_min, width, height]
        image_region = image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
        image_region = transform.resize(image_region, (224, 224))
        output = model(image_region)



