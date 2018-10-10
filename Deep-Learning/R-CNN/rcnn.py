# -*- coding:utf-8 -*-

import alex_net
import skimage.io
import skimage.transform as transform
import data_loader
import matplotlib.pyplot as plt
import selectivesearch
import torch


if __name__ == "__main__":

    # train model with ground_truth and region_proposals
    model = alex_net.train()
    torch.save(model.state_dict(), './alex_model.pkl')
    # predict image
    image = skimage.io.imread("./lena.jpg")
    img, regions = selectivesearch.selectivesearch(image)
    input = []
    for region in regions:
        r = region['rect']
        image_region = image[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        image_region = transform.resize(image_region, (224, 224))
        input.append(image_region)
    output = model(input)
    output = torch.max(output, axis = 1)
    # NMS
