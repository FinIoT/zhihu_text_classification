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
        fetches=model.y_pred
        feed_dict={model.X1_inputs:X1_batch,model.X2_inputs:X2_batch, model.y_inputs:y_batch,
                   model.batch_size:_batch_size, model.tst:True, model.keep_prob:1.0}
        predict_labels=sess.run(fetches,feed_dict)#batch_size*1999
        predict_scores.append(predict_labels)
        predict_labels=map(lambda label: lable.argsort()[-1:-6:-1], predict_labels)#batch_size*5
        predict_label_list.extend(predict_labels)
    predict_label_and_marked_label_list=zip(predict_label_list,marked_lable_list)
    precision, recall, f1=score_eval(predict_label_and_marked_label_list)
    print('Local valid p=%g, r=%g, f1=%g' % (precision, recall, f1))
    predict_scores=np.vstack(np.asarray(predict_scores))
    local_scores_name=local_scores_path+model.name+'/.npy'
    np.save(local_scores_name,predict_scores)
    print('local_scores.shape=', predict_scores.shape)
    print('Writed the scores into %s, time %g s' % (local_scores_name, time.time() - time0))
    
    
    
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