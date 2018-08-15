va# -*- coding: utf-8 -*-
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
        self.update_emas=list()
        
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
    def textcnn(self,inputs,filter_sizes):
        inputs_expand=tf.expand_dims(inputs,-1)#N*30*256*1
        pooled_outputs=[]
        for i,filtersize in enumerate(filter_sizes):
            with tf.name_scope('conv_max_%s'%filter_size):
                #卷积核的最后一维是卷积核个数
                filter_shape=[filtersize,self.embedding_size,1,self.n_filter]
                W_filter=tf.Varibale(tf.truncated_normal(filter_shape,stddev=0.1),name='W_filter')
                beta=tf.Varible(tf.constant(0.1,tf.float32,shape=self.n_filter),name='beta')
                tf.summary.histogram('beta',beta)
                #cnn三部曲：卷积（即线性），（BN）激活（非线性），池化（采集最大特征）
                conv=tf.nn.con2d(inputs_expand,W_filter,strides=[1,1,1,1],padding='VALID',name='conv')
                conv_bn,update_ema=self.batchnorm(conv,beta，convolutional=True)
                h=tf.nn.relu(conv_bn,name='relu')
                
                pooled=tf.nn.max_pool(h,ksize=[1,self.title_len-filter_size+1,1,1],strides=[1,1,1,1],
                                      padding='VALID',name='max_pool')
                pooled_outputs.append(pooled)#N*1*1*n_filter
                self.update_emas.append(update_emas)
        h_pool=tf.concat(pooled_outputs,3)#N*1*1*(n_filter*len(filter_sizes))
        n_filter_total=self.n_filter*len(filter_sizes)
        h_pool_flat=tf.reshape(h_pool,[-1,n_filter_total])
    
    def cnn_inference(self, X_inputs):
        inputs=tf.nn.embedding_lookup(self.embedding,X_inputs)#N*30*1024

        with tf.variable_scope('title_encoder'):
            title_outputs=self.textcnn()
        return title_outputs
        
        
        
        
        
        
        