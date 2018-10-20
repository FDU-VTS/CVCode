# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import torch
from torch import nn

class RoI_Pooling(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoI_Pooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.pooling = nn.AdaptiveMaxPool2d((pooled_height, pooled_width))

    def forward(self, features, rois):
        batch = rois.size()[0]
        num_rois = rois.size()[1]
        feature_h, feature_w = tuple(features.size()[2:])
        outputs_dim = (batch, num_rois) + tuple(features.size()[1:-2]) + (self.pooled_height, self.pooled_width, )
        outputs = torch.zeros(outputs_dim, device = "cpu", dtype=torch.double)
        for batch_index in range(batch):
            for roi_index, roi in enumerate(rois[batch_index]):
                cls, c, r, w, h = roi
                x = self.spatial_scale * float(c)
                y = self.spatial_scale * float(r)
                width = max(self.spatial_scale * float(w), 1)
                height = max(self.spatial_scale * float(h), 1)
                batch_size, rois_size, height_size, width_size = features.size()

                if y>= feature_h:
                    y = feature_h-height
                if x>= feature_w:
                    x = feature_w-width
                
                out = self.pooling(features[batch_index, :, int(y):int(y + height), int(x):int(x + width)])
                outputs[batch_index, roi_index, ..., :] = out
        return outputs

