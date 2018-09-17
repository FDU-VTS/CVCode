# -*- coding: utf-8 -*-

# Copyright (C) 2018 VTS, FUDAN UNIVERSITY
# @author Li Wenxi
# Efficient Graph-Based Image Segmentation[J]. International Journal of Computer Vision, 2004, 59(2):167-181.
# Felzenszwalb P F, Huttenlocher D P.

import cv2
import sys
import numpy as np


def diff(img, x1, y1, x2, y2):
    r = np.square(img[0][y1, x1] - img[0][y2, x2])
    g = np.square(img[1][y1, x1] - img[1][y2, x2])
    b = np.square(img[2][y1, x1] - img[2][y2, x2])
    return np.sqrt(r + b + g)


class universe():
    def __init__(self, elements):
        self.num = elements;
        self.elts = []
        for i in range(elements):
            rank = 0;
            size = 1;
            p = i;
            self.elts.append((rank, size, p))

    # a older func
    # def find(self, x):
    #     y = x
    #     while (y != self.elts[y][2]):
    #         y = self.elts[y][2]
    #     self.elts[x] = (self.elts[x][0], self.elts[x][1], y);
    #     return y

    # a newer func use recursion
    def find(self, u):
        if self.elts[u][2] == u:
            return u

        self.elts[u] = (self.elts[u][0], self.elts[u][1], self.find(self.elts[u][2]))
        return self.elts[u][2]

    def join(self, x, y):
        if self.elts[x][0] > self.elts[y][0]:
            self.elts[y] = (self.elts[y][0], self.elts[y][1], self.elts[x][2]);
            self.elts[x] = (self.elts[x][0], self.elts[x][1] + self.elts[y][1], self.elts[x][2])
        else:
            self.elts[x] = (self.elts[x][0], self.elts[x][1], self.elts[y][2])
            self.elts[y] = (self.elts[y][0], self.elts[y][1] + self.elts[x][1], self.elts[y][2])
            if self.elts[x][0] == self.elts[y][0]:
                self.elts[y] = (self.elts[y][0] + 1, self.elts[y][1], self.elts[y][2])
        self.num -= 1

    def size(self, x):
        return self.elts[x][1]

    def num_sets(self):
        return self.num


def THRESHOLD(size, c):
    return c / size

# Segment a graph
#
# Returns a disjoint-set forest representing the segmentation.
#
# num_vertices: number of vertices in graph.
# num_edges: number of edges in graph
# edges: array of edges.
# c: constant for treshold function.
def segment_graph(num_vertices, num_edges, graph, c):
    # make a disjoint-set forest
    u = universe(num_vertices)

    # init thresholds
    threshold = np.zeros(num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = THRESHOLD(1, c)

    # for each edge, in non-decreasing weight order...
    for i in range(num_edges):
        a = u.find(graph[i][0])
        b = u.find(graph[i][1])
        if a != b:
            if ((graph[i][2] <= threshold[a]) and
                    graph[i][2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = graph[i][2] + THRESHOLD(u.size(a), c)
    return u


def random_rgb():
    r = np.random.rand() * 255
    g = np.random.rand() * 255
    b = np.random.rand() * 255
    return (r, g, b)

# Segment an image
#
# Returns a color image representing the segmentation.
#
# im: image to segment.
# sigma: to smooth the image.
# c: constant for treshold function.
# min_size: minimum component size (enforced by post-processing stage).
# num_ccs: number of connected components in the segmentation.

def segment_image(im, sigma, c,
                  min_size, num_ccs):
    height, width, channel = im.shape
    im = np.array(im, dtype=float)
    gaussian_img = cv2.GaussianBlur(im, (5, 5), sigma)
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)
    # print(height, width, channel)

    # build graph
    graph = []
    num = 0;

    print("staring segment image")
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                a = y * width + x
                b = y * width + (x + 1)
                # w = 1
                w = diff(smooth_img, x, y, x + 1, y)
                num += 1
                graph.append((a, b, w))

            if y < height - 1:
                a = y * width + x
                b = (y + 1) * width + x
                w = diff(smooth_img, x, y, x, y + 1)
                num += 1
                graph.append((a, b, w))

            if x < width - 1 and y < height - 1:
                a = y * width + x
                b = (y + 1) * width + (x + 1)
                w = diff(smooth_img, x, y, x + 1, y + 1)
                num += 1
                graph.append((a, b, w))

            if x < width - 1 and y > 0:
                a = y * width + x
                b = (y - 1) * width + (x + 1)
                w = diff(smooth_img, x, y, x + 1, y - 1)
                num += 1
                graph.append((a, b, w))
        # print(x, y)

    # sort edges by weight
    # graph.sort(key=lambda x: (x[2]))
    graph = sorted(graph, key=lambda x: (x[2]))
    # segment

    u = segment_graph(width * height, num, graph, c)

    # post process small components
    for i in range(num):
        a = u.find(graph[i][0])
        b = u.find(graph[i][1])
        if (a != b) and ((u.size(a) < min_size) or u.size(b) < min_size):
            u.join(a, b)

    num_ccs.append(u.num_sets())

    colors = []
    for i in range(width * height):
        colors.append(random_rgb())

    print("staring random colors")

    # print("width", width, "height", height)

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            gaussian_img[y][x] = colors[comp]

    return gaussian_img


def main():
    if len(sys.argv) != 6:
        print("usage:", sys.argv[0], "sigma k min input output")
        return 1
    sigma = float(sys.argv[1])
    k = float(sys.argv[2])
    min_size = int(sys.argv[3])

    print("loading input image.")

    input = cv2.imread(sys.argv[4])

    print("processing")

    num_ccs = [];
    cv2.imwrite(sys.argv[5], segment_image(input, sigma, k,
                                           min_size, num_ccs))

    print("got", num_ccs[0], "components")
    print("done! uff...thats hard work.");

    return 0


if __name__ == '__main__':
    main()
