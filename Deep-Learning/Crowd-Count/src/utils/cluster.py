from scipy.io import loadmat
import h5py
import glob
import os
import re
import numpy as np


class ImgInfo:

    def __init__(self, image_name, number):
        self.image_name = image_name
        self.number = number


def get_index(avgs, img_info):
    number = img_info.number
    result = [abs(avg - number) for avg in avgs]
    return result.index(min(result))


def get_avg(arr):
    sum = 0
    for img_info in arr:
        sum += img_info.number
    return sum / len(arr)


def get_cluster(gt_path="../../data/shtu_dataset/original/part_A_final/train_data/ground_truth"):
    gts = glob.glob(os.path.join(gt_path, "*.mat"))
    p = re.compile("GT_(.*).mat")
    dic = []
    for gt in gts:
        mat = loadmat(gt)["image_info"]
        number = mat[0, 0][0, 0][1][0, 0]
        image_name = p.findall(gt)[0]
        img_info = ImgInfo(image_name, number)
        dic.append(img_info)

    cluster = np.array([[[200], []], [[800], []], [[1300], []]])
    for i in range(50):
        for img_info in dic:
            indics = get_index(cluster[:, 0], img_info)
            cluster[indics, 1].append(img_info)
        if i != 49:
            for i in range(3):
                cluster[i, 0] = get_avg(cluster[i, 1])
                cluster[i, 1] = []

    for i in range(3):
        print("avg is %d"%(cluster[i,0]))
        print([(img_info.image_name, img_info.number) for img_info in cluster[i, 1]])

    return cluster


cluster = get_cluster()