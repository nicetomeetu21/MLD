# -*- coding:utf-8 -*-
# basic 2d datasets
import os
import cv2 as cv
from torch.utils.data.dataset import Dataset
import numpy as np
from natsort import natsorted
import torch
import torchvision
from torchvision import transforms
import albumentations as A


class AugmentedDataset(Dataset):
    def __init__(self, dataroot, label_root, cube_names=None, ret_info=False):

        self.ret_info=ret_info
        self.bscan_paths = []
        self.bscan_names = []
        self.cube_names = []
        self.label_pathes = []
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize((0.5,),(0.5,))
        ])
        if cube_names is None:
            cube_names = natsorted(os.listdir(os.path.join(dataroot)))
        for cube_name in cube_names:
            img_names = natsorted(os.listdir(os.path.join(dataroot, cube_name)))

            for img_name in img_names:
                self.bscan_paths.append(os.path.join(dataroot, cube_name, img_name))
                self.label_pathes.append(os.path.join(label_root, cube_name, img_name))
                self.bscan_names.append(img_name)
                self.cube_names.append(cube_name)


    def __getitem__(self, index):
        img = cv.imread(self.bscan_paths[index], cv.IMREAD_GRAYSCALE)
        label = cv.imread(self.label_pathes[index], cv.IMREAD_GRAYSCALE)
        # img = self.transform(image=img, masks = [label])['image']
        if self.transform is not None:
            transformed = self.transform(image=img, masks=[label])
            img = transformed['image']
            label = transformed['masks'][0]

        img = transforms.ToTensor()(img)
        label = transforms.ToTensor()(label)
        if self.ret_info:
            return img,label, self.cube_names[index], self.bscan_names[index]
        else :
            return img,label

    def __len__(self):
        return len(self.bscan_paths)


class AugmentedDataset_test(Dataset):
    def __init__(self, vessel_root, layer_root, cube_names=None, ret_info=False, skip_stride = 1):

        self.ret_info=ret_info
        self.vessel_paths = []
        self.layer_paths = []
        self.bscan_names = []
        self.cube_names = []

        if cube_names is None:
            cube_names = natsorted(os.listdir(os.path.join(vessel_root)))
        for cube_name in cube_names:
            img_names = natsorted(os.listdir(os.path.join(vessel_root, cube_name)))

            for img_name in img_names:
                self.vessel_paths.append(os.path.join(vessel_root, cube_name, img_name))
                assert os.path.exists(os.path.join(layer_root, cube_name, img_name))
                self.layer_paths.append(os.path.join(layer_root, cube_name, img_name))
                self.bscan_names.append(img_name)
                self.cube_names.append(cube_name)
        self.vessel_paths = self.vessel_paths[::skip_stride]
        self.layer_paths = self.layer_paths[::skip_stride]

        self.cube_names = self.cube_names[::skip_stride]

        self.bscan_names = self.bscan_names[::skip_stride]
    def __getitem__(self, index):
        img = cv.imread(self.vessel_paths[index], cv.IMREAD_GRAYSCALE)
        region = cv.imread(self.layer_paths[index], cv.IMREAD_GRAYSCALE)

        img = transforms.ToTensor()(img)
        region = transforms.ToTensor()(region)
        if self.ret_info:
            return img, region, self.cube_names[index], self.bscan_names[index]
        else :
            return img, region

    def __len__(self):
        return len(self.vessel_paths)
