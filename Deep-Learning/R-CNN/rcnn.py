# -*- coding:utf-8 -*-

import alex_net
import skimage.io
import selectivesearch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import data_loader
import utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == "__main__":

    # train model with ground_truth and region_proposals
    model = alex_net.train()
    torch.save(model.state_dict(), './alex_model.pkl')

    # get image region proposal with selective search
    image = skimage.io.imread("./test_data/000012.jpg")
    img, regions = selectivesearch.selective_search(image)
    input = []

    # reduce noise
    candidate = []
    for region in regions:
        if region['rect'] in candidate:
            continue
        else:
            if region['size'] < 10:
                continue
        candidate.append(region['rect'])

    # get image regions
    for r in candidate:
        image_region = image[r[1]:r[1] + r[3] + 1, r[0]:r[0] + r[2] + 1]
        input.append(image_region)

    # get test data to tensors
    tsfm = transforms.Compose([data_loader.TestRescale((224, 224)), data_loader.TestToTensor()])
    test_data_loader = data_loader.TestLoader(input, transform=tsfm)
    test_data = torch.utils.data.DataLoader(test_data_loader, batch_size=1, shuffle=True, num_workers=2)
    result = []
    for data in iter(test_data):
        data = data.float()
        output = model(data)
        output = torch.max(output, 1)
        result.append(output)
    result = np.asarray(result)
    input = np.asarray(input).reshape(-1, 1)
    candidate = np.asarray(candidate).reshape(-1, 4)
    nms_sum = np.concatenate((candidate, input, result), axis=1)

    # NMS
    regions = utils.NMS(nms_sum)

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)
    for x, y, w, h, label in regions:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        plt.annotate(label, xy=(x, y))

    plt.show()



