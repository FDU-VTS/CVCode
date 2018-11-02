# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------

"""Compress a Fast R-CNN network using truncated SVD."""


import torch

def compress_weights(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = torch.svd(W, some=True)
    Ul = U[:, :l]
    sl = s[:l]
    Vl = V[:l, :]

    L = torch.dot(torch.diag(sl), Vl)
    return Ul, L