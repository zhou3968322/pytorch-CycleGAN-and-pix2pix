# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/11/25
import numpy as np
import cv2
import math
from generator.constant_aug import seq_rec, seq_squ, seq_cir, seq_cir_big, seq_ell


def aug_square(image, segmentation_maps=None):
    """
    image: h, w, c, c=1 or 3
    segmentation_maps: mask
    """
    if segmentation_maps is None:
        return seq_squ(image=image)
    else:
        return seq_squ(image=image, segmentation_maps=segmentation_maps)


def aug_rectangle(image, segmentation_maps):
    return seq_rec(image=image, segmentation_maps=segmentation_maps)


def aug_circle(image, segmentation_maps, random_state=0.1):
    if random_state < 0.1:
        return seq_cir_big(image=image, segmentation_maps=segmentation_maps)
    else:
        return seq_cir(image=image, segmentation_maps=segmentation_maps)


def aug_ellipse(image, segmentation_maps=None, random_state=0.5):
    if random_state < 0.4:
        return seq_cir(image=image, segmentation_maps=segmentation_maps)
    elif random_state < 0.5:
        return seq_cir_big(image=image, segmentation_maps=segmentation_maps)
    else:
        return seq_ell(image=image, segmentation_maps=segmentation_maps)


def get_seg_map_ellipse(im, save_path=None):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    # cnt = contours[1][0]

    nonb = thresh.nonzero()
    cnt = np.array([[[nonb[1][i], nonb[0][i]]] for i in range(len(nonb[0]))])

    # 最小外接矩形框，有方向角
    y = [item[0][0] for item in cnt]
    x = [item[0][1] for item in cnt]
    minx = min(x)
    miny = min(y)
    maxx = max(x)
    maxy = max(y)
    points = [[miny, minx], [miny, maxx], [maxy, maxx], [maxy, minx]]
    points = np.int0(points)
    if save_path is not None:
        cv2.drawContours(im, [points], 0, (255, 0, 255), 2)
        cv2.imwrite(save_path, im)
    return points.tolist()


def get_seg_map_rectangle(im, save_path=None):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    # cnt = contours[1][0]

    nonb = thresh.nonzero()
    cnt = np.array([[[nonb[1][i], nonb[0][i]]] for i in range(len(nonb[0]))])

    # 最小外接矩形框，有方向角
    rect = cv2.minAreaRect(cnt)
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    if save_path is not None:
        cv2.drawContours(im, [points], 0, (255, 0, 255), 2)
        cv2.imwrite(save_path, im)
    return points.tolist()


def get_seg_map_circle(im, save_path=None):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    # cnt = contours[1][0]
    nonb = thresh.nonzero()
    cnt = np.array([[[nonb[1][i], nonb[0][i]]] for i in range(len(nonb[0]))])

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if save_path is not None:
        cv2.circle(im, center, radius, (255, 0, 0), 2)
        cv2.imwrite(save_path, im)
    points = []
    for angle in range(0, 360, 20):
        points.append([
            center[0] + radius * math.cos(angle * math.pi / 180), center[1] + radius * math.sin(angle * math.pi / 180)
        ])

    return points