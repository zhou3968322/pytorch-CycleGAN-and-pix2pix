# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/11/25
from imgaug import augmenters as iaa


seq_cir = iaa.Sequential(
    [
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        # iaa.MultiplyElementwise((0.8, 0.99)),
        iaa.Dropout(p=(0, 0.05)),
        # iaa.JpegCompression(compression=(80, 99)),
        iaa.Affine(rotate=(-90, 90), scale=(0.4, 0.7), fit_output=True)
    ],
    random_order=True)

seq_cir_big = iaa.Sequential(
    [
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        # iaa.MultiplyElementwise((0.8, 0.99)),
        iaa.Dropout(p=(0, 0.05)),
        # iaa.JpegCompression(compression=(80, 99)),
        iaa.Affine(rotate=(-90, 90), scale=(0.9, 1.5), fit_output=True)
    ],
    random_order=True)


seq_ell = iaa.Sequential(
    [
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        # iaa.MultiplyElementwise((0.8, 0.99)),
        iaa.Dropout(p=(0, 0.05)),
        # iaa.JpegCompression(compression=(80, 99)),
        iaa.Affine(rotate=(-20, 20), scale=(0.4, 0.9), fit_output=True)
    ],
    random_order=True)


seq_squ = iaa.Sequential(
    [
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        # iaa.MultiplyElementwise((0.8, 0.99)),
        iaa.Dropout(p=(0, 0.05)),
        # iaa.JpegCompression(compression=(80, 99)),
        iaa.Affine(rotate=(-90, 90), scale=(0.18, 0.35), fit_output=True)
        # iaa.Affine(rotate=(-90, 90), scale=(0.8, 1.4), fit_output=True)
    ],
    random_order=True)


seq_rec = iaa.Sequential(
    [
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        # iaa.MultiplyElementwise((0.8, 0.99)),
        iaa.Dropout(p=(0, 0.05)),
        # iaa.JpegCompression(compression=(80, 99)),
        iaa.Affine(rotate=(-90, 90), scale=(0.15, 0.25), fit_output=True)
        # iaa.Affine(rotate=(-90, 90), scale=(0.2, 0.4), fit_output=True)
    ],
    random_order=True)

seq_doc_noise = iaa.Sequential(
    [
        iaa.Sometimes(
            0.6,
            iaa.OneOf(iaa.Sequential([iaa.GaussianBlur(sigma=(0, 1.0))])
                                      # iaa.AverageBlur(k=(2, 5)),
                                      # iaa.MedianBlur(k=(3, 7))])
            )
        ),
        iaa.Sometimes(
            0.5,
            iaa.LinearContrast((0.8, 1.2), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.3,
            iaa.Multiply((0.8, 1.2), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.3,
            iaa.WithBrightnessChannels(iaa.Add((-40, 40))),
        ),
        # iaa.Sometimes(
        #     0.3,
        #     iaa.OneOf(iaa.Sequential([
        #                 iaa.AdditiveGaussianNoise(scale=(0, 0.01*255), per_channel=0.5),
        #                 iaa.SaltAndPepper(0.01)]))
        # ),
        iaa.Sometimes(
            0.5,
            iaa.Add((-10, 10), per_channel=0.5),
        ),
        # iaa.Sometimes(
        #     0.5,
        #     iaa.Dropout(p=(0, 0.05))
        # ),
        # iaa.JpegCompression(compression=(80, 99))
    ],
    random_order=True)
