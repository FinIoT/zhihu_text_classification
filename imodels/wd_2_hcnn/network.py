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
        self.update_emas=list()
        self._global_step=tf.Variable(0,trainable=False,name='Global_step')
        #placeholders
        self._tst=tf.placeholder(tf.bool)
        self._keep_prob=tf.placeholder(tf.float32,[])
        self._batch_size=tf.placeholder(tf.int32,[])
        
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
        with tf.variable_scope('fc_bn_layer'):
            output=tf.concat([output_title,output_content],axis=1)
            fc_shape=[self.n_filter*(len(self.sent_fliter_sizes)+len(self.doc_filter_sizes)),self.fc_hidden_size]
            fc_W=self.weight_variable(fc_shape,'fc_Weight')
            tf.summary.histogram('fc_W',fc_W)
            fc_bias=self.bias_variable(self.fc_hidden_size,'fc_bias')
            tf.summary.histogram('fc_bias',fc_bias)
            fc_results=tf.matmul(output,fc_W,name='fc_h')
            fc_bn,update_ema_fc=self.batchnorm(fc_results,fc_bias)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu=tf.nn.relu(fc_bn,name='relu')
            fc_bn_drop=tf.nn.dropout(self.fc_bn_relu,self._keep_prob)
            
        with tf.varialbe_scope('out_layer'):
            out_W=self.weight_variable([self.fc_hidden_sizes,self.n_class],name='out_W')
            tf.summary.histogram('out_W',out_W)
            out_b=self.bias_variable([self.n_class],name='out_b')
            tf.summary.histogram('out_b',out_b)
            self._y_pred=tf.nn.xw_plus_b(fc_bn_drop,out_W,out_b,name='y_pred')
            
        with tf.name_scope('loss'):
            self._loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred,
                                                                              labels=self._y_inputs))
            tf.summary.scalar('loss',self._loss)
        self.saver=tf.train.Saver(max_to_keep=2)
    @property
    def global_step(self):
        return self._global_step
    @property
    def tst(self):
        return self._tst
    @property
    def keep_prob(self):
        return self._keep_prob
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def X1_inputs(self):
        return self._X1_inputs
    
    @property
    def X2_inputs(self):
        return self._X1_inputs
    @property
    def y_inputs(self):
        return self._y_inputs
    @property
    def y_pred(self):
        return self._y_pred
    def loss(self):
        return self._loss
    
    def weight_variable(self,shape,name):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name=name)
    def bias_variable(self,shape,name):
        return tf.Variable(tf.constant(0.1,shape=shape),name=name)
    
    
    def batchnorm(self, Ylogits, offset, convolutional=False):
        exp_moving_avg=tf.train.ExponentialMovingAverage(0.999,self._global_step)
        bnepsilon=1e-5
        
        if convolutional:
            mean,variance=tf.nn.moments(Ylogits,[0,1,2])#数据形状变为[256,]，也就是只剩下256个卷积核的均值方差了
        else: 
            mean,variance=tf.nn.moments(Ylogits,[0])
        
        update_moving_avg=exp_moving_avg.apply([mean,variance])
        m=tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v=tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn=tf.nn.batch_normalization(Ylogits,m,v,offset,None,bnepsilon)
        return Ybn,update_moving_avg
    
    
    def textcnn(self,inputs,n_step,filter_sizes,embed_size):
        inputs_expand=tf.expand_dims(inputs,-1)#N*30*256*1
        pooled_outputs=[]
        for i,filtersize in enumerate(filter_sizes):
            with tf.name_scope('conv_max_%s'%filter_size):
                #卷积核的最后一维是卷积核个数
                filter_shape=[filtersize,embed_size,1,self.n_filter]
                W_filter=tf.Varibale(tf.truncated_normal(filter_shape,stddev=0.1),name='W_filter')
                beta=tf.Varible(tf.constant(0.1,tf.float32,shape=self.n_filter),name='beta')
                tf.summary.histogram('beta',beta)
                #cnn三部曲：卷积（即线性），（BN）激活（非线性），池化（采集最大特征）
                conv=tf.nn.con2d(inputs_expand,W_filter,strides=[1,1,1,1],padding='VALID',name='conv')
                conv_bn,update_ema=self.batchnorm(conv,beta，convolutional=True)
                h=tf.nn.relu(conv_bn,name='relu')
                
                pooled=tf.nn.max_pool(h,ksize=[1,n_step-filter_size+1,1,1],strides=[1,1,1,1],
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
    def hcnn_inference(self,X_inputs):
        
        
        inputs=tf.reshape(X_inputs,[self.batch_size*self.doc_len,self.sent_len])
        inputs_embed=tf.nn.embedding_lookup(self.embedding,inputs)
        #生成句子向量
        sent_outputs=self.textcnn(input_embed,self.sent_len,self.sent_filter_sizes,self.embedding_size)
        doc_inputs=tf.reshape(sent_outputs,[self.bath_size, self.doc_len, 
                                            self.n_filter*len(self.sent_filter_sizes)])
        #生成文档向量
        doc_outputs=self.textcnn(doc_inputs,self.doc_len,self.doc_filter_sizes,
                                 self.n_filter*len(self.sent_filter_sizes))
        #shape=self.batch_size,self.n_filters*len(self.doc_filter_sizes)
        return doc_outputs
        
        
        
        
        
        