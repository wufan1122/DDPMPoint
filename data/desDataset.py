
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import os.path
import cv2

IMG_FOLDER_NAME = "images"
LIST_IMG_NAME = 'list_img'

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_path(root_dir, split, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, split, img_name)


class desDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.res = resolution
        self.data_name_len = data_len
        self.split = split

        self.root_dir = dataroot
        self.split = split  # train | val | test

        self.list_img_path = os.path.join(self.root_dir, LIST_IMG_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_img_path)

        self.dataset_img_name_len = len(self.img_name_list)
        if self.data_name_len <= 0:
            self.data_name_len = self.dataset_img_name_len
        else:
            self.data_name_len = min(self.data_name_len, self.dataset_img_name_len)


    def __len__(self):
        return self.data_name_len

    def __getitem__(self, index):
        img_name = self.img_name_list[index % self.data_name_len]
        img_path = get_img_path(self.root_dir, self.split, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(index + 1)
        img1 = Util.transform_augment_cd(img, split=self.split, min_max=(-1, 1))

        return {'img1': img1, 'index': index}