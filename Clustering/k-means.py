# -*- coding:utf-8 -*-

#   the most important is the distance between 2 clusters
#   and the average value in a cluster
#   in this implement, distance is a - b
#   average is sum()/number
#   break condition is k * threshold


import numpy as np


# data_set: data need to be clustered
# k: how many types to be wanted
# linear data is supposed
def k_means(data_set, k, threshold):
    data_set = np.array(data_set).astype(np.int)
    size = data_set.shape[0]
    cluster_indices = np.random.choice(data_set, k, replace=False)
    cluster = dict(zip(cluster_indices, cluster_indices))
    # convert value to list
    for k, v in cluster.items():
        cluster[k] = [v]
    cluster_old = cluster
    num = 0
    while num < 50:
        for data in data_set:
            # get the closest cluster of data
            key = np.array([k for k, v in cluster.items()])
            diff = np.abs(key - data)
            index = np.where(diff == min(diff))[0][0]
            temp_key = key[index]
            # change value and then change key
            avg = (sum(cluster[temp_key]) + data) / (len(cluster[temp_key]) + 1)
            cluster[temp_key].append(data)
            cluster[avg] = cluster.pop(temp_key)
        # if difference between cluster and cluster_old < 6, break
        cluster_key = np.array([k for k, v in cluster.items()])
        cluster_oldkey = np.array([k for k, v in cluster_old.items()])
        if np.sum(cluster_key - cluster_oldkey) < threshold * k:
            break
        num += 1
    return cluster

if __name__ == "__main__":
    a = np.random.random_sample(1000) * 50
    cluster = k_means(a, 30, 1)
    for k, v in cluster.items():
        print(k, v)

