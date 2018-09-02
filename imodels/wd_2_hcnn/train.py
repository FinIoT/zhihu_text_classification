# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:27:23 2018

@author: LIKS
"""

import tensorflow as tf
import network
import os
import shutil
import numpy as np

flags=tf.flags
flags.DEFINE_bool('is_retrain',True,'if is_retrain is true, rebuild the summary')
FLAGS=flags.FLAGS

settings=network.Settings()
summary_path = settings.summary_path
ckpt_path=settings.ckpt_path

embedding_path = '../../data/word_embedding.npy'


def main(_):
    ##summary and ckpt path
    global ckpt_path
    global last_f1
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    elif FLAGS.is_retrain:
        shutil.rmtree(summary_path)
        os.makedirs(summary_path)
    
    print('start to import embeddings...')
    W_embedding=np.load(embedding_path)
    print('embedding imported.') 
    
    print('Building the model')
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #引入模型
        model=network.HCNN(settings,W_embedding) 
        ##optimizer tf.train.AdamOptimizer(lr=..).minimize(loss)
        with tf.variable_scope('training_ops') as vs:
            learning_rate=tf.train.exponential_decay(FLAGS.lr,model.global_step,FLAGS.decay_step,
                                                    FLAGS.decay_rate,staircase=True)
        #two optimizer, one updates the embedding, one not
            with tf.variable_scope('optimizer1'):
                tvar1=tf.trainable_variables()
                grads1=tf.gradients(model.loss,tvar1)
                optimizer1=tf.train.AdamOptimizer(learning_rate=learning_rate)
                op1=optimizer1.apply_gradients(zip(grads1,tvar1),global_step=model.global_step)
            with tf.variable_scope('optimizer2'):
                #
                tvar2=[tvar for tvar in tvar1 if 'embedding' not in tvar.name]
                grads2=tf.gradients(model.loss,tvar2)
                optimizer2=tf.train.AdamOptimizer(learning_rate=learning_rate)
                op2=optimizer2.apply_gradients(zip(grads2,tvar2),global_step=model.global_step)
            update_op=tf.group(*model.update_emas)
            merged=tf.summary.merge_all()
            train_writer=tf.summary.FileWriter(summary_path+'train', sess.graph)
            test_writer=tf.summary.FileWriter(summary_path+'test')
            #感觉这句是多余的，它收集不到training_ops变量名下的其它变量啊，换句话说这是个空列表。
            training_ops=[v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]
            
        ##如果已经保存模型导入上次的
        if os.path.exists(summary_path+'checkpoint')
        #开始epoch循环训练
        ##满足条件则保存模型           
        #引入title,content数据,数据形式应该是：N*30，N*300

if __name__=='__main__':
    tf.app.run()
    