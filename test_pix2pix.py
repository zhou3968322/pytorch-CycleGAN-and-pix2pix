# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/11/24

"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from options.test_options import TestOptions
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im
import os
from PIL import Image
import torch
from data.base_dataset import default_transform


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.name = "document_pix2pix"
    opt.model = "pix2pix"
    opt.netG = "unet_256"
    opt.direction = "AtoB"
    opt.norm = "batch"
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
    # img_path = "test/input/343abe6f-9e28-4813-b87a-cdb098f8c17f_page1.jpg"
    # img_path = "test/input/保利文化集团股份有限公司2019年第三季度财务报表-3.jpg"
    # img_path = "test/input/常州天宁建设发展集团有限公司2019年三季度合并及母公司财务报表-0.jpg"
    # img_path = "test/input/常州市晋陵投资集团有限公司2019年三季度合并及母公司财务报表-0.jpg"
    # img_path = "test/input/重庆市万盛经济技术开发区开发投资集团有限公司2019年三季度财务报表-1.jpg"
    img_path = "test/input/重庆市万盛经济技术开发区开发投资集团有限公司2019年三季度财务报表-2.jpg"
    res_img_dir = "test/output"
    signet_offsets = [(200, 100, 712, 612),
                      (500, 100, 1012, 612),
                      (200, 100, 712, 612),
                      (200, 100, 712, 612),
                      (1000, 200, 1512, 712),
                      (1500, 100, 2012, 612)]
    os.makedirs(res_img_dir, exist_ok=True)
    img_path_list = [img_path]
    import time
    for img_path in img_path_list:
        img = Image.open(img_path).convert("RGB")
        a_img = img.crop(signet_offsets[5])
        t0 = time.time()
        transform = default_transform()
        A = transform(a_img)
        A = torch.unsqueeze(A, dim=0)
        A = A.to(torch.device("cuda:0"))
        fake_B = model.netG(A)
        res_B = tensor2im(fake_B)
        image_pil = Image.fromarray(res_B)
        print("cost:{}".format(time.time() - t0))
        out_img_path = os.path.join(res_img_dir, os.path.basename(img_path))
        image_pil.save(out_img_path)

