# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 21:50:18 2018

@author: LIKS
"""

import tensorflow as tf

class Settings():
    def __init__(self):
        self.model_name='wd_2_hcnn'
        self.title_len=self.sent_len=30
        self.doc_len=10
        self.sent_filter_sizes=[2,3,4,5]
        self.doc_filter_sizes=[2,3,4]
        self.n_filter=256
        self.fc_hidden_size=1024
        self.n_class=1999
        self.summary_path='../../summary/'+self.model_name+'/'
        self.ckpt_path='../../ckpt/'+self.model_name+'/'

class HCNN():
    """
    title: inputs->embedding->textcnn->output_title
    content: inputs->embeding->hcnn->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """
    def __init__(self, settings,W_embedding):
        self.model_name = settings.model_name
        self.sent_len=self.title_len = settings.sent_len
        self.doc_len = settings.doc_len
        self.sent_filter_sizes = settings.sent_filter_sizes
        self.doc_filter_sizes = settings.doc_filter_sizes
        self.n_filter = settings.n_filter
        self.n_class = settings.n_class
        self.fc_hidden_size=settings.fc_hidden_size
        
        with tf.name_scope('inputs'):
            self._X1_inputs=tf.placeholder(dtype=tf.int64,shape=[None,self.title_len],name='X1_inputs')#N*30
            #shape是句子长度乘以句子数目
            self._X2_inputs=tf.placeholder(dtype=tf.int64,shape=[None,self.doc_len*self.sent_len],name='X2_inputs')
            self._y_inputs=tf.placeholder(dtype=tf.int64,shape=[None,self.n_class],name='y_inputs')
        
        with tf.variable_scope('embedding'):
            #这里可以直接用self.embedding=W_embedding代替吗？不可以，应该将矩阵转化为张量，并且后期可能要训练W_embedding
            #直接self.embedding=W_embedding以后就无法训练了
            self.embedding=tf.get_variable(name='embedding',shape=W_embedding.shape,
                                           initializer=tf.constant_initializer(W_embedding),trainable=True)
        self.embedding_size=W_embedding.shape[1]
        with tf.variable_scope('cnn_text'):
            #换成self.X1_inputs也可以，但是self.X1_inputs一般是供外部访问的.类内部尽量用私有变量
            output_title=self.cnn_inference(self._X1_inputs)
        with tf.variable_scope('hcnn_content'):
            output_content=self.hcnn_inference(self._X2_inputs)
        
    @property
    def X1_inputs(self):
        return self._X1_inputs
    
    @property
    def X2_inputs(self):
        return self._X1_inputs
    def textcnn(self,inputs,filter_sizes,):
        inputs_expand=tf.expand_dims(inputs,-1)#N*30*1024*1
        for i in filter_sizes:
            shape=[1,i,self.embedding_size]
    
    def cnn_inference(self, X_inputs):
        inputs=tf.nn.embedding_lookup(self.embedding,X_inputs)#N*30*1024

        with tf.variable_scope('title_encoder'):
            title_outputs=self.textcnn()
        return title_outputs
        
        
        
        
        
        
        