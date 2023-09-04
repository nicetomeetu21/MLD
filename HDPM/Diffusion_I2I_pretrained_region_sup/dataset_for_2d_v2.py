# -*- coding:utf-8 -*-
import os

import torch
from natsort import natsorted
from torch.utils.data.dataset import Dataset
import cv2 as cv
from torchvision import transforms
import numpy as np

def get_A_paths_with_list(A_root, cube_names):

    A_paths = []
    rela_dirs = []
    names = []
    for i, (cube_name) in enumerate(cube_names):
        assert os.path.exists(os.path.join(A_root, cube_name))
        img_names = natsorted(os.listdir(os.path.join(A_root, cube_name)))
        for img_name in img_names:
            A_paths.append(os.path.join(A_root, cube_name, img_name))
            rela_dirs.append(cube_name)
            names.append(img_name)

    return A_paths, rela_dirs, names

class OCT2D_multi_Augmented_Dataset(Dataset):
    def __init__(self, data_roots, with_path = False, transform = None):
        data_root = data_roots[0]
        assert os.path.exists(data_root)
        cube_names = natsorted(os.listdir(data_root))
        _, cube_names, img_names = get_A_paths_with_list(data_root, cube_names)

        self.data_roots = data_roots
        self.cube_name_list = cube_names
        self.img_name_list = img_names

        self.transform = transform
        self.with_path = with_path

    def __getitem__(self, index):
        img_list = []
        for data_root in self.data_roots:
            img =  cv.imread(os.path.join(data_root, self.cube_name_list[index],self.img_name_list[index]), cv.IMREAD_GRAYSCALE)
            img_list.append(img)

        if self.transform is not None:
            if len(img_list) > 1:
                transformed = self.transform(image=img_list[0],  masks=img_list[1:])
                img_list[0] = transformed['image']
                img_list[1:] = transformed['masks']
            else:
                transformed = self.transform(image=img_list[0])
                img_list[0] = transformed['image']

        for i in range(len(img_list)):
            img_list[i] = transforms.ToTensor()(img_list[i])

        if self.with_path:
            return img_list, self.cube_name_list[index], self.img_name_list[index]
        else:
            return img_list

    def __len__(self):
        return len(self.img_name_list)



class OCT2D_multi_npy_Augmented_Dataset(Dataset):
    def __init__(self, data_roots, npy_roots, with_path = False, transform = None):
        data_root = data_roots[0]
        assert os.path.exists(data_root)
        cube_names = natsorted(os.listdir(data_root))
        _, cube_names, img_names = get_A_paths_with_list(data_root, cube_names)

        self.data_roots = data_roots
        self.npy_roots = npy_roots
        self.cube_name_list = cube_names
        self.img_name_list = img_names

        self.transform = transform
        self.with_path = with_path

    def __getitem__(self, index):
        img_list = []
        for data_root in self.data_roots:
            img =  cv.imread(os.path.join(data_root, self.cube_name_list[index],self.img_name_list[index]), cv.IMREAD_GRAYSCALE)
            img_list.append(img)

        for npy_root in self.npy_roots:
            img = np.load(os.path.join(npy_root, self.cube_name_list[index],self.img_name_list[index][:-4]+'.npy'))
            img_list.append(img)

        if self.transform is not None:
            if len(img_list) > 1:
                transformed = self.transform(image=img_list[0],  masks=img_list[1:])
                img_list[0] = transformed['image']
                img_list[1:] = transformed['masks']
            else:
                transformed = self.transform(image=img_list[0])
                img_list[0] = transformed['image']

        for i in range(len(img_list)):
            img_list[i] = transforms.ToTensor()(img_list[i])

        if self.with_path:
            return img_list, self.cube_name_list[index], self.img_name_list[index]
        else:
            return img_list

    def __len__(self):
        return len(self.img_name_list)