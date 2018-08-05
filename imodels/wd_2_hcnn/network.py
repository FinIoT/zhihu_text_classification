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
            self._X1_inputs=tf.placeholder(dtype=tf.int64,shape=[None,self.title_len],name='X1_inputs')
            #shape是句子长度乘以句子数目
            self._X2_inputs=tf.placeholder(dtype=tf.int64,shape=[None,self.doc_len*self.sent_len],name='X2_inputs')
        
        
@property
def X1_inputs(self):
    return self._X1_inputs

        
        
        
        
        
        
        