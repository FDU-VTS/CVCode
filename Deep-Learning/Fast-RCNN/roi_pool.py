# -*- coding:utf-8 -*-
from torch.autograd import Function
import torch
import math
<<<<<<< HEAD

=======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
>>>>>>> 33bfc40e934cef7448e12a536b796a1e44e5d553

def roi_pooling_forward(pooled_height, pooled_width, spatial_scale, features, rois):
    num_rois, size_rois = rois.size()
    assert features.size(0) == 1
    _, num_channels, data_height, data_width = torch.tensor(features.size()).tolist()
<<<<<<< HEAD
    argmax = torch.IntTensor(num_rois, num_channels, pooled_height, pooled_width, 2).zero_()
    output = torch.zeros(num_rois, num_channels, pooled_height, pooled_width)
=======
    argmax = torch.IntTensor(num_rois, num_channels, pooled_height, pooled_width, 2, device=DEVICE).zero_()
    output = torch.zeros(num_rois, num_channels, pooled_height, pooled_width, device=DEVICE)
>>>>>>> 33bfc40e934cef7448e12a536b796a1e44e5d553
    for roi_index in range(len(rois)):
        roi = rois[roi_index]
        # roi positions
        # x_min, y_min, x_max, y_max mean position on feature map
        x, y, w, h = roi
        x_min = x * spatial_scale
        y_min = y * spatial_scale
        x_max = (x + w) * spatial_scale
        y_max = (y + h) * spatial_scale
        # get bins
        w_bin = w * spatial_scale / pooled_width
        h_bin = h * spatial_scale / pooled_height
        for ph in range(pooled_height):
            for pw in range(pooled_width):
                # init position
                h_start = math.floor(ph * h_bin)
                w_start = math.floor(pw * w_bin)
                h_end = math.ceil((ph + 1) * h_bin)
                w_end = math.ceil((pw + 1) * w_bin)
                # avoid position out of boundary
                h_start = int(min(max(h_start + y_min, 0), data_height - 1))
                w_start = int(min(max(w_start + x_min, 0), data_width - 1))
                h_end = int(min(max(h_end + y_min, 0), data_height - 1))
                w_end = int(min(max(w_end + x_min, 0), data_width - 1))
                # avoid boundary is empty
                if h_end <= h_start or w_end <= w_start:
                    output[roi_index, :, ph, pw] = 0
                    argmax[roi_index, :, ph, pw] = torch.IntTensor([-1, -1])
                else:
                    # for each channel, set the pooling result
                    # and get the argmax position
                    for channel_index in range(num_channels):
                        bin_area = features[0, channel_index, h_start:h_end + 1, w_start:w_end + 1]
                        pool_result = torch.max(bin_area)
                        pool_index = torch.argmax(bin_area)
                        pool_argmax = torch.IntTensor([h_start + math.floor(pool_index / (w_end - w_start + 1)),
                                                       w_start + math.floor(pool_index % (w_end - w_start + 1))])
                        output[roi_index, channel_index, ph, pw] = pool_result
                        argmax[roi_index, channel_index, ph, pw] = pool_argmax
    return output, argmax


class ROIPool(Function):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None
        self.argmax = None

    # rois: (n, pos)
    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        output, argmax = roi_pooling_forward(self.pooled_height, self.pooled_width,
                                             self.spatial_scale, features, rois)
        self.argmax = argmax
<<<<<<< HEAD
        print("argmax's size is: ", argmax.size())
=======
>>>>>>> 33bfc40e934cef7448e12a536b796a1e44e5d553

        return output

    def backward(self, grad_output):
<<<<<<< HEAD
        print("grad_output's size is: ", grad_output.size())
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width)
        print("grad_input's size is: ", grad_input.size())
=======
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width, device=DEVICE)
>>>>>>> 33bfc40e934cef7448e12a536b796a1e44e5d553
        for i in range(self.rois.size(0)):
            for k in range(num_channels):
                for m in range(self.pooled_height):
                    for n in range(self.pooled_width):
                        x, y = self.argmax[i, k, m, n]
                        if(y == -1 or x == -1):
                            continue
                        grad_input[:, k, x, y] = grad_output[i, k, m, n]

<<<<<<< HEAD
        return grad_input, None
=======
        return grad_input, None
>>>>>>> 33bfc40e934cef7448e12a536b796a1e44e5d553
