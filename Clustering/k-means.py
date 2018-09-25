# -*- coding:utf-8 -*-

import numpy as np


# data_set: data need to be clustered
# k: how many types to be wanted
# linear data is supposed
def k_means(data_set, k):
    data_set = np.array(data_set)
    size = data_set.shape[0]
    cluster_indices = np.random.choice(data_set, k, replace=False)
    cluster = dict(zip(cluster_indices, cluster_indices))
    cluster_old = cluster
    num = 0
    while num < 50:
        for data in data_set:
            # get the closest cluster of data
            min_value = np.argwhere(min(np.abs(np.array(v for k, v in cluster) - data))).reshape(-1)
            # update cluster
            cluster[min_value[0]] = (sum(cluster[min_value[0]]) + data) / len(cluster[min_value[0]]) + 1
        if cluster == cluster_old:
            break
        num += 1
    return cluster



