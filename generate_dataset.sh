# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/11/28

# test
PYTHONPATH=/data/zhoubingcheng/project/pytorch-CycleGAN-and-pix2pix \
  python datasets/make_crop_seal_dataset.py \
  --data_dir /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_signet_aligned_data \
  --out_data_dir /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_crop_signet_data_1 \
  --random_seed 20 --num_workers 0 --max_img_num 4 --aug_times 2

#generate
PYTHONPATH=/data/zhoubingcheng/project/pytorch-CycleGAN-and-pix2pix \
  python datasets/make_crop_seal_dataset.py \
  --data_dir /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_signet_aligned_data \
  --out_data_dir /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_crop_signet_data \
  --random_seed 24 --num_workers 64 --aug_times 2



# generate from pdf

PYTHONPATH=/data/zhoubingcheng/project/pytorch-CycleGAN-and-pix2pix \
 python datasets/make_aug_seal_dataset_from_pdf.py \
 --random_seed 15 --num_workers 64 --train_pdf_sum 11000 --val_pdf_sum 1000 --test_pdf_sum 120
