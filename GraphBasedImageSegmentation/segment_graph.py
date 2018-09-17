# -*- coding:utf-8 -*-

from disjoint_set import *


def THRESHOLD(size,c):
    return c/size


def segment_graph(num_vertices,num_edges,edges,c):
    edges = sorted(edges,key=lambda edges:edges[2])
    print(len(edges))
    u = universe(num_vertices)
    threshold = [0 for i in range(num_vertices)]
    for i in range(num_vertices):
        threshold[i] = THRESHOLD(1,c)
    for i in range(num_edges):
        pedge = edges[i]
        a = u.find(pedge[0])
        b = u.find(pedge[1])

        if a != b:
            if pedge[2] <= threshold[a] and pedge[2] <= threshold[b]:
                u.join(a,b)
                a = u.find(a)
                threshold[a] = pedge[2] + THRESHOLD(u.get_size(a),c)

    return u

