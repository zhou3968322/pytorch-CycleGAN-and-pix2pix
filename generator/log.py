# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/11/25

import logging, os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s\t%(levelname)s\t%(name)s '
                           '%(filename)s:%(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt='%(asctime)s\t%(levelname)s\t%(name)s '
                                      '%(filename)s:%(lineno)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


def set_log_path(log_path):
    if os.path.isfile(log_path):
        r_handler = logging.FileHandler(log_path, "a", 'utf-8')
    else:
        r_handler = logging.FileHandler(log_path, "w", "utf-8")
    logger.addHandler(r_handler)


