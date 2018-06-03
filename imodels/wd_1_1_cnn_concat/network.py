# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:45:45 2018

@author: LIKS
"""

import tensorflow as tf

"""wd_1_1_cnn_concat
title 部分使用 TextCNN；content 部分使用 TextCNN； 两部分输出直接 concat。
"""

class Settings():
    def __init__(self):
        self.model_name='wd_1_1_cnn_concat'
        self.title_len=30
        self.content_len=150
        self.filter_sizes=[2,3,4,5,7]
        self.n_filter=256
        self.fc_hidden_size=1024
        self.n_class=1999
        self.summary_path='../../summary/' + self.model_name + '/'
        self.ckpt_path='../../ckpt/'+self.model_name+'/'