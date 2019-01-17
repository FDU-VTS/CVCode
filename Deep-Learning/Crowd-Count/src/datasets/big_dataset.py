from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import cv2
import numpy as np
import h5py
import skimage.io
import skimage.color
import scipy.io as io


class BigDataset(Dataset):

    def __init__(self, mode="train", **kwargs):
        self.big_list = self.get_big_data()
        self.root = "./data/shtu_dataset/original/part_A_final/train_data/" if mode == "train" else \
                "./data/shtu_dataset/original/part_A_final/test_data/"
        self.temp = glob.glob(self.root + "images/*.jpg")
        self.paths = []
        for img_path in self.temp:
            if img_path in self.big_list:
                self.paths.append(img_path)
        if mode == "train":
            self.paths *= 4
        self.transform = kwargs['transform']
        self.length = len(self.paths)
        self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img, den = self.dataset[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, den

    def load_data(self):
        result = []
        index = 0
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = Image.open(img_path).convert('RGB')
            gt_file = h5py.File(gt_path)
            den = np.asarray(gt_file['density'])
            h = den.shape[0]
            w = den.shape[1]
            h_trans = h // 8
            w_trans = w // 8
            den = cv2.resize(den, (w_trans, h_trans),
                             interpolation=cv2.INTER_CUBIC) * (h * w) / (h_trans * w_trans)
            result.append([img, den])
            if index % 100 == 99 or index == self.length - 1:
                print("load {0}/{1} images".format(index + 1, self.length))
            index += 1
        return result

    def get_big_data(self):
        big_root = './data/shtu_dataset/original/'
        part_A_train = os.path.join(big_root, 'part_A_final/train_data', 'images')
        part_A_test = os.path.join(big_root, 'part_A_final/test_data', 'images')
        path_sets = [part_A_train, part_A_test]
        big_list = []

        for path in path_sets:
            for img_path in glob.glob(os.path.join(path, '*.jpg')):
                mat = io.loadmat(
                    img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
                number = mat["image_info"][0, 0][0, 0][1]
                if number[0, 0] >= 400:
                    big_list.append(img_path)

        return big_list