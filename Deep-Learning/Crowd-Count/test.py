import skimage.io
from scipy.io import loadmat
import numpy as np

path = "./data/original/part_A_final/train_data/images/IMG_1.jpg"
sum = 0
for index in range(1, 301):
    print(index)
    gt = "./data/original/part_B_final/train_data/ground_truth/GT_IMG_{0}.mat".format(index)
    image = skimage.io.imread(path)
    mat = loadmat(gt)
    image_info = mat["image_info"]
    ann_points = image_info[0][0][0][0][0]
    d_mean = 0.0
    for i in ann_points:
        y = (ann_points - i) * (ann_points - i)
        y = np.sum(y, axis=1)
        y = np.sort(y)
        y = np.sqrt(y)
        mean = np.sum(y[:4]) / 3
        d_mean += mean
    d_mean /= len(ann_points)
    sum += d_mean
print(sum / 300)