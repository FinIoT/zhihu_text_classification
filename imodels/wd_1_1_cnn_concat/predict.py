# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:58:33 2018

@author: LIKS
"""
import tensorflow as tf
import numpy as np
import os

import network

settings=network.Settings()
ckpt_path=settings.ckpt_path
embedding_path = '../../data/word_embedding.npy'

def main(_):
    #引入模型-先判断需要引入的路径存不存在
    if not os.path.exists(ckpt_path+'checkpoint'):
        print('there is no saved model, exit the program')
        exit()
    print('loading the model')
    W_embedding=np.load(embedding_path)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #建立Session,feed数据,run
    with tf.Session(config=config) as sess:
        model=network.TextCNN(W_embedding,settings)
        model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        print('Local predicting...')
        local_predict(sess,model)
        print('Test predicting')
        predict(sess,model)
        
        
    

    

if __name__=='__main__':
   tf.app.run()