# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/11/25
import subprocess
import sys, os

if "GS_COMMAND" in os.environ:
    GS = os.environ["GS_COMMAND"]
else:
    GS = "gs"


def pdf2img(pdf_file, prefix, dpi=320):
    gs_command = "{} -sDEVICE=jpeg " \
                 "-dSAFER -dBatch -r{} -o {}_page%d.jpeg {}".format(GS, dpi, prefix, pdf_file)
    process = subprocess.Popen(gs_command.split(), stdout=sys.stdout, stderr=sys.stderr)
    process.communicate()

    img_path_list = []
    i = 1
    jpeg_path = "{}_page{}.jpeg".format(prefix, i)
    while os.path.isfile(jpeg_path):
        img_path_list.append(jpeg_path)
        i += 1
        jpeg_path = "{}_page{}.jpeg".format(prefix, i)
    return img_path_list
