# -*- coding:utf-8 -*-
import os
from natsort import natsorted
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
import cv2 as cv
import numpy as np
def reduce_layers(masks, cnt=4):
    h,w = masks.shape
    # print(masks.shape)
    for j in range(w):
        axes = np.argwhere(masks[ :,j]==255)
        # print(axes)
        top = axes[0][0]
        bottom = min(axes[0][0]+cnt,axes[-1][0])
        # print(top, bottom)
        # exit()
        masks[top:bottom+1,j] = 0
    return masks

def solve_cube(src_dir, region_mask_dir, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    names = natsorted(os.listdir(src_dir))
    for name in names:
        img = cv.imread(os.path.join(src_dir, name), cv.IMREAD_GRAYSCALE)
        region_mask = cv.imread(os.path.join(region_mask_dir, name), cv.IMREAD_GRAYSCALE)
        thresh_niblack = threshold_niblack(img, window_size=25, k=0.5)

        result = img > thresh_niblack
        result = result.astype(np.uint8)
        result = 1-result
        result *= 255

        region_mask = reduce_layers(region_mask, cnt=5)
        region_mask = region_mask.astype(bool)
        result *= region_mask
        cv.imwrite(os.path.join(result_dir, name), result)


def solve_cubes(src_dir, region_mask_dir, result_dir):
    cube_names = natsorted(os.listdir(src_dir))
    for name in cube_names:
        solve_cube(os.path.join(src_dir, name),os.path.join(region_mask_dir, name),os.path.join(result_dir, name))
if __name__ == '__main__':
    src_dir = r'the dir to OCT cubes'
    region_mask_dir = r'the dir to region mask'
    result_dir = r'the dir to result cubes'
    solve_cubes(src_dir, region_mask_dir, result_dir)