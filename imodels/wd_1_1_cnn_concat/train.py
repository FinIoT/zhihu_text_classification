# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:28:42 2018

@author: LIKS
"""

"""
训练模型的存放ckpt，tensorboard数据的存放
网络模型
导入数据
"""
import tensorflow as tf
import os,shutil

def main():
    global ckpt_path
    global last_f1
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    elif not FLAGS.is_retain:
        shutil.rmtree(summary_path)
        os.makedirs(summary_path)

if __name__ == '__main__':
    tf.app.run()