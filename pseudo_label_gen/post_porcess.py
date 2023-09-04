# -*- coding:utf-8 -*-
import os
from natsort import natsorted
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
import cv2 as cv
import numpy as np
from scipy.interpolate import splprep, splev
def smooth_contours(contours):
    smoothened = []
    # s = set()
    for contour in contours:

        # print(contour[0][0][0],contour[0][0][1])
        # exit()
        # for i in range(len(contour)):
        #     if (contour[i] == contour[i-1]).all():
        #         del contour[i]
        # if (contour[0] == contour[- 1]).all():
        #     del contour[-1]
        #     s.add((p[0][0],p[0][1]))
        # for p in s:
        # contour = list(set(contour))
        x,y = contour.T
        # print(contour)
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)[0]
        # x_new = np.take_along_axis(x, okay, axis=0)
        # print(okay)
        x_new = []
        y_new = []
        for i in okay:
            # print(i)
            x_new.append(x[i])
            y_new.append(y[i])
        x = np.r_[x_new, x[-1], x[0]]
        y = np.r_[y_new, y[-1], y[0]]
        # print(x)
        # print(y)
        if len(x) < 5 or len(y) < 5:
            smoothened.append(contour)
            continue
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x,y], s=1.0, per=True)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 15)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))
    return smoothened
def elipse_fit(img, contours):

    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))
    ret = np.zeros_like(img)
    for contour, area in zip(contours, areas):
        if area < np.percentile(areas, 75):
            ret = cv.fillPoly(ret,[contour], 255)
        else:
            ellipse = cv.fitEllipse(contour)
            # ret = cv.fillPoly(ret, pts =[contour], color=(255))
            ret = cv.ellipse(ret, ellipse, 255, thickness=-1)
    return ret
def elipse_fit2(img, contours):

    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))
    ret = np.zeros_like(img)
    for contour, area in zip(contours, areas):
        if area < np.percentile(areas, 50):
            ret = cv.fillPoly(ret,[contour], 255)
        else:
            ret2 = np.zeros_like(img)
            ret2 = cv.fillPoly(ret2,[contour], 255)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
            ret2 = cv.dilate(ret2, kernel, iterations=1)
            #
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
            ret2 = cv.erode(ret2, kernel, iterations=1)
            ret += ret2
    return ret
def convx_fit(img, contours):
    hulls = []
    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))
        hulls.append(cv.convexHull(contour, False))
    for hull, area in zip(hulls, areas):
        if area > np.percentile(areas, 25):
            img = cv.fillPoly(img,[hull], 255)


    # kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # # img = cv.erode(img, kernel, iterations=1)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)
    return img

def solve_cube(src_dir, region_mask_dir, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    names = natsorted(os.listdir(src_dir))
    for name in names:
        img = cv.imread(os.path.join(src_dir, name), cv.IMREAD_GRAYSCALE)



        kernel_size = 5
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

        img = (img>127).astype(np.uint8)*255
        os.makedirs(os.path.join(os.path.join(result_dir,'GaussianBlur')),exist_ok=True)
        cv.imwrite(os.path.join(result_dir,'GaussianBlur', name), img)


        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
        img = cv.erode(img, kernel, iterations=1)
        #
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img = cv.dilate(img, kernel,iterations=1)

        os.makedirs(os.path.join(os.path.join(result_dir,'MORPH_ELLIPSE')),exist_ok=True)
        cv.imwrite(os.path.join(result_dir,'MORPH_ELLIPSE', name), img)


        # edged = cv.Canny(img, 30, 200)
        contours, hierarchy = cv.findContours(img,
                                               cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # print(len(contours))

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
        # img = cv.erode(img, kernel, iterations=1)
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
        # img = cv.dilate(img, kernel, iterations=1)
        # kernel_size = 5
        # img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
        #
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # img = cv.dilate(img, kernel,iterations=1)
        #


        ret = convx_fit(img, contours)
        result = ret
        os.makedirs(os.path.join(os.path.join(result_dir,'convx_fit')),exist_ok=True)
        cv.imwrite(os.path.join(result_dir,'convx_fit', name), result)


def solve_cubes(src_dir, region_mask_dir, result_dir):
    cube_names = natsorted(os.listdir(src_dir))
    for name in cube_names:
        if name != '10164':continue
        solve_cube(os.path.join(src_dir, name),os.path.join(region_mask_dir, name),os.path.join(result_dir, name))
        # break
if __name__ == '__main__':
    src_dir = r''
    region_mask_dir = r''
    result_dir = r''
    solve_cubes(src_dir, region_mask_dir, result_dir)