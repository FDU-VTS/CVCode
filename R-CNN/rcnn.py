# -*- coding:utf-8 -*-

import alex_net
import region_proposals
import torchvision.transforms as transforms


if __name__ == "__main__":

    net = alex_net.train()
    regions = region_proposals.image_proposal("./lane.jpg")
    # resize regions
    transforms.Resize()
    # predict