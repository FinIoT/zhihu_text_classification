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
        
class TextCNN():
        """
    title: inputs->textcnn->output_title
    content: inputs->textcnn->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """
    def __init__(self, W_embedding, setting):
        self.model_name = settings.model_name
        self.title_len = settings.title_len
        self.content_len = settings.content_len
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)#256*5
        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size
        self._global_step=tf.Variable(0,trainable=False,name='Global_Step')
        self.update_emas=list()
        #placeholders
        self._tst=tf.placeholder(tf.bool)
        self._keep_prob=tf.placeholder(tf.float32,[])
        self._batch_size=tf.placeholder(tf.int32,[])
        
        
        with tf.name_scope('Inputs'):
            self._X1_inputs=tf.placeholder(tf.int64,[None,self.title_len],name='X1_inputs')
            self._X2_inputs=tf.placeholder(tf.int64,[None,self.content_len],name='X2_inputs')
            self._y_inputs=tf.placeholder(tf.float32,[None,self.n_class],name='y_inputs')
        with tf.variable_scope('embedding'):
            self.embedding=tf.get_variable(name='embedding',shape=W_embedding,
                                           initializer=tf.constant_initializer(W_embedding),trainable=True)
        self.embedding_size=W_embedding.shape[1]#1024
        
        with tf.variable_scope('cnn_text'):
            output_title=self.cnn_inference(self._X1_inputs, self.title_len)
        with tf.variable_scope('cnn_content'):
            output_content=self.cnn_inference(self._X2_inputs, self.content_len)
            
    def weight_variable(self,shape,name):    
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial,name=name)
    def bias_variable(self,shape,name):    
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial,name=name)        
    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batchnormalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        exp_moving_avg=tf.train.ExponentialMovingAverage(0.999,self._global_step)
        bnepsilon=1e-5
        if convolutional:
            #以某一维度计算均值或方差
            mean,variance=tf.moment(Ylogits,[0,1,2])
        


        
        
        
    def cnn_inference(self,X_inputs,n_step):
         """TextCNN 模型。
        Args:
            X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
            title_outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        inputs=tf.nn.embedding_lookup(self.embedding, X_inputs)#batch_size, n_step,embedding_size
        inputs=tf.expand_dims(input,-1)#batch_size, n_step,embedding_size,1
        pooled_outputs=list()
        for i,filter_size in enumerate(self.filter_sizes):
            filter_shape=[filter_size,self.embedding_size,1,self.n_filter]
            W_filter=self.weight_variable(shape=filter_shape,name='W_filter')
            beta=self.bias_variable(shape=[self.n_filter],name='beta_filter')
            tf.summary.histogram('beta',beta)
            conv=tf.nn.conv2d(inputs,W_filter,strides=[1,1,1,1],padding='VALID',name='conv')+beta
            #激活层前面加BN
            conv_bn,update_emas=self.batchnorm(conv,beta,convolutional=True)
        
        
        
        
        
        
        
        