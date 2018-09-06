# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:27:23 2018

@author: LIKS
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

flags = tf.flags
flags.DEFINE_bool('is_retrain', False, 'if is_retrain is true, not rebuild the summary')
flags.DEFINE_integer('max_epoch', 1, 'update the embedding after max_epoch, default: 1')
flags.DEFINE_integer('max_max_epoch', 6, 'all training epoches, default: 6')
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_float('decay_rate', 0.65, 'decay rate, default: 0.65')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob for training, default: 0.5')
flags.DEFINE_integer('decay_step', 1000, 'decay_step, default: 1000')
flags.DEFINE_integer('valid_step', 500, 'valid_step, default: 500')
flags.DEFINE_float('last_f1', 0.10, 'if valid_f1 > last_f1, save new model. default: 0.10')
FLAGS = flags.FLAGS

lr = FLAGS.lr
last_f1 = FLAGS.last_f1
settings = network.Settings()
title_len = settings.title_len
summary_path = settings.summary_path
ckpt_path = settings.ckpt_path
model_path = ckpt_path + 'model.ckpt'

embedding_path = '../../data/word_embedding.npy'
data_train_path = '../../data/wd-data/seg_train/'
data_valid_path = '../../data/wd-data/seg_valid/'
tr_batches = os.listdir(data_train_path)  # batch 文件名列表
va_batches = os.listdir(data_valid_path)
n_tr_batches = len(tr_batches)
n_va_batches = len(va_batches)

def get_batch(data_path,batch_id):
    new_batch=np.load(data_path+str(batch_id)+".npz")
    X_batch=new_batch['X']
    y_batch=new_batch['y']
    X1_batch=X_batch[:,:title_len]
    X2_batch=X_batch[:,title_len:]
    return [X1_batch,X2_batch,y_batch]

def valid_epoch(data_path,sess,model):
    va_batches=os.listdir(data_path)
    n_va_batches=len(va_batches)
    _costs=0.0
    predict_labels_list=list()
    marked_labels_list=list()
    for i in range(n_va_batches):
        [X1_batch,X2_batch,y_batch]=get_batch(data_path,i)
        marked_labels_list.extend(y_batch)
        y_batch=to_categorical(y_batch)
        _batch_size=len(y_batch)
        fetches=[model.loss,model.y_pred]
        feed_dict={model.X1_inputs:X1_batch, model.X2_inputs:X2_batch, model.y_inputs:y_batch,
                   model.batch_size:_batch_size, model.tst:True, model.keep_prob:1.0}
        _cost,predict_labels=sess.run(fetches,feed_dict)
        _costs+=_cost
        predict_labels=map(lambda label:label.argsort()[-1:-6:-1], predict_labels)
        predict_labels_list.extend(predict_labels)
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)
    mean_cost = _costs / n_va_batches
    return mean_cost, precision, recall, f1
    

def train_epoch(data_path, sess, model, train_fetches, valid_fetches, train_writer, test_writer):
    #获取数据
    global last_f1
    time0=time.time()
    batch_index=np.random.permutation(n_tr_batches)
    for batch in tqdm(range(n_tr_batches)):
        global_step=sess.run(model.global_step)
        if 0==(global_step+1)%FLAGS.valid_step:
            valid_cost, precision, recall, f1 = valid_epoch(data_valid_path, sess, model)
            print('Global_step=%d: valid cost=%g; p=%g, r=%g, f1=%g, time=%g s' % (
                global_step, valid_cost, precision, recall, f1, time.time() - time0))
            time0 = time.time()
            if f1>last_f1:
                last_f1=f1
                saving_path=model.saver.save(sess,model_path,global_step+1)
                print('saved new model to %s'%saving_path)
        [X1_batch, X2_batch, y_batch]=get_batch(data_train_path,batch_index[batch])
        y_batch=to_categorical(y_batch)
        _batch_size = len(y_batch)
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
                     model.batch_size: _batch_size, model.tst: False, model.keep_prob: FLAGS.keep_prob}
        summary, _cost, _, _ = sess.run(train_fetches, feed_dict)
        #valid per 500 step
        if (global_step+1)%500==0:
            train_writer.add_summary(summary,global_step)
            batch_id=np.random.int(0,n_tr_batches)
            [X1_batch,X2_batch,y_batch]=get_batch(batch_id)
            y_batch = to_categorical(y_batch)
            _batch_size = len(y_batch)
            feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
            summary, _cost = sess.run(valid_fetches, feed_dict)
            test_writer.add_summary(summary, global_step)                            
            
            
        
    
    #喂数据

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
            
        ##如果已经保存模型导入上次的模型，初始化相关变量
        if os.path.exists(ckpt_path+'checkpoint'):
            print('Restoring Variables from checkpoint...')
            model.saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))
            last_valid_cost, precision, recall, last_f1 = valid_epoch(data_valid_path, sess, model)
            print('valid_cost=%g, p=%g, r=%g, f1=%g'%(last_valid_cost,precision,recall,last_f1))
            sess.run(tf.variables_initializer(training_ops))
            train_op2=train_op1
        else:
            print('start to initialize the variables...')
            sess.run(tf.global_variables_initializer())
        #开始epoch循环训练
        print('3.begin training...')
        print('max_epoch=%d, max_max_epoch=%d'%(FLAGS.max_epoch, FLAGS.max_max_epoch))
        for epoch in range(FLAGS.max_max_epoch):
            global_step=sess.run(model.global_step)
            print('global step d%, lr=%g'%(global_step,sess.run(learning_rate)))
            if epoch==FLAGS.max_epoch:
                train_op=op1
            else:
                train_op=op2
            train_fetches=[merged,model.loss,train_op,update_op]
            valid_fetches=[merged,model.loss]
            train_epoch(data_train_path,sess,model,train_fetches,valid_fetches,train_writer,test_writer)
        ##满足条件则保存模型 
        valid_cost,precision,recall,f1=valid_epoch(data_valid_path,sess,model) 
        print('END.Global_step=%d: valid cost=%g; p=%g, r=%g, f1=%g' % (
                sess.run(model.global_step), valid_cost, precision, recall, f1)) 
        if f1>last_f1:
            saving_path=model.saver.save(sess,model_path,sess.run(model.global_step)+1)
            print('saved new model to:%s'%saving_path)
        #引入title,content数据,数据形式应该是：N*30，N*300

if __name__=='__main__':
    tf.app.run()
    