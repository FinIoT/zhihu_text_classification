# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:37:12 2018

@author: LIKS
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import sys
import os

sys.path.append('../')
from data_helpers import pad_X30
from data_helpers import pad_X52
from data_helpers import wd_pad_cut_docs
from data_helpers import ch_pad_cut_docs
from data_helpers import train_batch
from data_helpers import eval_batch

wd_train_path = '../data/wd-data/seg_train/'
wd_valid_path = '../data/wd-data/seg_valid/'
wd_test_path = '../data/wd-data/seg_test/'
ch_train_path = '../data/ch-data/seg_train/'
ch_valid_path = '../data/ch-data/seg_valid/'
ch_test_path = '../data/ch-data/seg_test/'
paths = [wd_train_path, wd_valid_path, wd_test_path,
         ch_train_path, ch_valid_path, ch_test_path]
for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)
#数据打包       
def wd