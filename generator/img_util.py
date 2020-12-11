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
from PIL import Image
from imgaug import augmenters as iaa
import random
import imgaug as ia


def resize_im(im, img_min_side=None, img_max_side=None, keep_small=False, method=Image.ANTIALIAS):
    """
    keep_small : 是否保持小图的大小
    """
    assert img_min_side is not None or img_max_side is not None
    h, w = im.size
    if img_min_side:
        short_edge = min(h, w)
        if short_edge <= img_min_side and keep_small:
            return im
        scale = img_min_side / short_edge
        im = im.resize((int(h * scale), int(w * scale)), method)
    else:
        long_edge = max(h, w)
        if long_edge <= img_max_side and keep_small:
            return im
        scale = img_max_side / long_edge
        im = im.resize((int(h * scale), int(w * scale)), method)
    return im


def random_rotate(im, degree):
    angle = degree
    return im.rotate(angle, Image.BICUBIC, expand=1)


def convert_img_to_blue_hsv(rgb_img=None, bgr_img=None):
    assert rgb_img is not None or bgr_img is not None
    if rgb_img is not None:
        bgr_img = rgb_img[:, :, ::-1]
    img_hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * np.random.normal(0.8, 0.05)
    img_hsv[:, :, 0] = img_hsv[:, :, 0] + np.random.normal(8, 3)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * np.random.normal(0.9, 0.02)
    return img_hsv


def convert_img_to_red_hsv(rgb_img=None, bgr_img=None):
    assert rgb_img is not None or bgr_img is not None
    if bgr_img is not None:
        rgb_img = bgr_img[:, :, ::-1]
    img_hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * np.random.normal(0.8, 0.05)
    img_hsv[:, :, 0] = img_hsv[:, :, 0] + np.random.normal(8, 3)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * np.random.normal(0.9, 0.02)
    return img_hsv


def convert_hsv_to_black_hsv(img_hsv):
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * np.random.normal(0.5, 0.01)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * np.random.normal(0.15, 0.02)
    return img_hsv


def generate_seq_aug_iaa_seq(rotate_angle, scale):
    """
    """
    return iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        # iaa.MultiplyElementwise((0.8, 0.99)),
        iaa.Dropout(p=(0, 0.05)),
        # iaa.JpegCompression(compression=(80, 99)),
        iaa.Affine(rotate=rotate_angle, scale=scale, fit_output=True)
        # iaa.Affine(rotate=(-90, 90), scale=(0.8, 1.4), fit_output=True)
    ],
        random_order=True)


def get_pil_img_random_mask(pil_img):
    point_thresh = random.uniform(1.8, 2)
    return pil_img.convert('L').point(lambda lut: lut * point_thresh)


def is_box_outside_rect(poly, rect):
    """
    poly 是否在rect外, true 为outside

    """
    poly = np.array(poly)
    xl, yt, xr, yd = rect
    if len(poly.shape) == 2:
        if poly[:, 0].max() <= xl or poly[:, 0].min() >= xr:
            return True
        if poly[:, 1].max() <= yt or poly[:, 1].min() >= yd:
            return True
        return False
    else:
        x0, y0, x1, y1 = poly
        if x1 <= xl or x0 >= xr:
            return True
        if y1 <= yt or y0 >= yd:
            return True
        return False


def is_box_inside_rect(poly, rect):
    """
    poly 是否完全在rect内, true 为outside

    """
    poly = np.array(poly)
    xl, yt, xr, yd = rect  # [xl, xr)左闭右开
    if len(poly.shape) == 2:
        if poly[:, 0].min() <= xl or poly[:, 0].max() >= xr:
            return False
        if poly[:, 1].min() <= yt or poly[:, 1].max() >= yd:
            return False
        return True
    else:
        x0, y0, x1, y1 = poly
        if x0 <= xl or x1 >= xr:
            return False
        if y0 <= yt or y1 >= yd:
            return False
        return True


def convert_rect_to_polygon(rect):
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x0, y1],
                     [x1, y1], [x1, y0]])


def convert_poly_to_rect(poly):
    poly = np.array(poly)
    x0 = np.min(poly[:, 0])
    x1 = np.max(poly[:, 0])
    y0 = np.min(poly[:, 1])
    y1 = np.max(poly[:, 1])
    return np.array([x0, y0, x1, y1])


def set_seeds(seed_num):
    random.seed(seed_num)
    ia.seed(seed_num)