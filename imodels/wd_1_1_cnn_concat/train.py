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
import numpy as np
from tqdm import tqdm
import os
import sys
import shutil
import time
import network

sys.path.append('../..')
from data_helpers import to_categorical
from evaluator import score_eval

flags=tf.flags
flags.DEFINE_bool('is_retain',False,'if is_retain is true, not rebuild the summary')
flags.DEFINE_integer('max_epoch', 1, 'update the embedding after max_epoch, default: 1')
flags.DEFINE_integer('max_max_epoch', 1, 'all training epoches, default: 6')
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_float('decay_rate', 0.65, 'decay rate, default: 0.65')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob for training, default: 0.5')
# 正式
flags.DEFINE_integer('decay_step', 15000, 'decay_step, default: 15000')
flags.DEFINE_integer('valid_step', 10000, 'valid_step, default: 10000')
flags.DEFINE_float('last_f1', 0.40, 'if valid_f1 > last_f1, save new model. default: 0.40')

# 测试
# flags.DEFINE_integer('decay_step', 1000, 'decay_step, default: 1000')
# flags.DEFINE_integer('valid_step', 500, 'valid_step, default: 500')
# flags.DEFINE_float('last_f1', 0.10, 'if valid_f1 > last_f1, save new model. default: 0.10')
FLAGS = flags.FLAGS

lr=FLAGS.lr
last_f1=FLAGS.last_f1
settings=network.Settings()
title_len=settings.title_len
summary_path=settings.summary_path
ckpt_path = settings.ckpt_path
model_path = ckpt_path + 'model.ckpt'

embedding_path = '../../data/word_embedding.npy'
data_train_path = '../../data/wd-data/data_train/'
data_valid_path = '../../data/wd-data/data_valid/'
tr_batches = os.listdir(data_train_path)  # batch 文件名列表(1.npz~)
va_batches = os.listdir(data_valid_path)
n_tr_batches = len(tr_batches)
n_va_batches = len(va_batches)

def main():
    global ckpt_path
    global last_f1
    #把不存在的ckpt，summary建立起来；重新训练时删除原来的summary
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    elif not FLAGS.is_retain:
        shutil.rmtree(summary_path)
        os.makedirs(summary_path)
    #引入数据--已经训练好的embedding,穿插打印些提示语言
    print('lodaing embedding...')
    W_embedding=np.load(embedding_path)
    print('training sample_num = %d' % n_tr_batches)#23360
    print('valid sample_num = %d' % n_va_batches)#782
    
    #构建模型，  
    print('building model...')
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    #打开Session,模型图必须在Session中构建，对Sesssion中的config进行设置
    with tf.Session() as sess:
        #构建模型图
        model=network.TextCNN(W_embedding,settings)
        with tf.variable_scope('training_ops') as vs:
            learning_rate=tf.train.exponential_decay(lr,model.global_step,FLAGS.decay_step,FLAGS.decay_rate
                                                     ,staircase=True)
            
            #two optimizer: Embedding not updated, and updated
            with tf.variable_scope('Optimizer1'):
                tvars1=tf.trainable_variables()
                grads1=tf.gradients(model.loss,tvars1)
                optimizer1=tf.train.AdamOptimizer(learing_rate=learning_rate)
                optimizer1.apply_gradients(zip(grads1,tvars1),global_step=model.global_step)
            with tf.varialbe_scope('Optimizer2'):
                tvars2=[tvar for tvar in tvars1 if 'embedding' not in tvar.name]
                grads2=tf.gradients(model.loss,tvars2)
                optimizer2=tf.train.AdamOptimizer(learning_rate=learning_rate)
                optimizer2.apply_gradients(zip(grads2,tvars2),global_step=model.global_step)
            update_op=tf.group(*model.update_emas)
            merged=tf.summary.merge_all()
            train_writer=tf.summary.FileWriter(summary_path+'train', sess.graph)
            test_writer=tf.summary.FileWriter(summary_path+'test',sess.graph)
            training_ops=[v for v in tf.global_variables() if v.name.starwit(vs.name+'/')]
        
        #若保存过模型，导入
        if os.path.exists(ckpt_path+'checkpoint'):
            print('restoring model from chekcpoint...')
            model.saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))
            last_valid_cost,precision, recall, last_f1=valid_epoch(data_valid_path,sess,model)
                
        
        #填充数据
        
    


    
        
        
if __name__ == '__main__':
    tf.app.run()