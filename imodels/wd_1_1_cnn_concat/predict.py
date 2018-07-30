# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:58:33 2018

@author: LIKS
"""
import tensorflow as tf
import numpy as np
import os
import time
import network
import tqdm

settings=network.Settings()
title_len=settings.title_len
ckpt_path=settings.ckpt_path

embedding_path = '../../data/word_embedding.npy'
data_valid_path = '../../data/wd-data/data_valid/'

va_batches=os.listdir(data_valid_path)
n_va_batches=len(va_batches)

def get_batch(batch_id):
    new_batch=np.load(data_valid_path+str(batch_id)+'.npz')
    X_batch=new_batch['X']
    y_batch=new_batch['y']
    X1_batch=X_batch[:,:title_len]
    X2_batch=X_batch[:,title_len:]
    return [X1_batch,X2_batch,y_batch]

def local_predict(sess, model):
    #test on the valid data
    time0=time.time()
    predict_label_list=list()
    marked_label_list=list()
    predict_scores=list()
    for i in tqdm(range(n_va_batches)):
        [X1_batch, X2_batch, y_batch]=get_batch(i)
        marked_label_list.extend(y_batch)
        _batch_size=len(X1_batch)
    
    
def predict(sess, model):
    

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