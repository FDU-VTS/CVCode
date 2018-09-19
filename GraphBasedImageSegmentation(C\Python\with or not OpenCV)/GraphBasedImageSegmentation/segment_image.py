# -*-coding:utf-8 -*-

from filter import *
import random
import numpy as np
from segment_graph import *
import matplotlib.pyplot as plt


# difference between 2 pixels
def diff(r,g,b,x1,y1,x2,y2):
    return np.sqrt(np.square(r[x1, y1] - r[x2, y2]) +
                   np.square(g[x1,y1] - g[x2,y2]) +
                   np.square(b[x1,y1] - b[x2,y2]))


def segment_image(image_origin,sigma,k,min_size,num_css):
    width = image_origin.shape[0]
    height = image_origin.shape[1]

    r = image_origin[:, :, 0]
    g = image_origin[:, :, 1]
    b = image_origin[:, :, 2]

    smooth_r = Filter.smooth(r, sigma)
    smooth_g = Filter.smooth(g, sigma)
    smooth_b = Filter.smooth(b, sigma)

    edges = []
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                x_pos = y * width + x
                y_pos = y * width + (x + 1)
                defference = diff(smooth_r,smooth_g,smooth_b,x,y,x+1,y)
                edges.append([x_pos,y_pos,defference])
                num += 1
            if y < height - 1:
                x_pos = y * width + x
                y_pos = (y + 1) * width + x
                defference = diff(smooth_r, smooth_g, smooth_b, x, y, x ,y+1)
                edges.append([x_pos, y_pos, defference])
                num += 1
            if x < width - 1 and y < height - 1:
                x_pos = y * width + x
                y_pos = (y + 1) * width + (x + 1)
                defference = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1)
                edges.append([x_pos, y_pos, defference])
                num += 1
            if x < width - 1 and y > 0:
                x_pos = y * width + x
                y_pos = (y-1) * width + (x + 1)
                defference = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1)
                edges.append([x_pos, y_pos, defference])
                num += 1
    u = segment_graph(width * height, num, edges, k)
    for i in range(num):
        a = u.find(edges[i][0])
        b = u.find(edges[i][1])
        if a != b and ((u.get_size(a) < min_size) or (u.get_size(b) < min_size)):
            u.join(a, b)
    num_css = u.num
    output = np.zeros((width,height,3))
    colors = []
    for i in range(width*height):
        colors.append([random.random()*255 for j in range(3)])

    for y in range(height):
        for x in range(width):
            comp = u.find(y*width + x)
            output[x,y] = colors[comp]

    return output