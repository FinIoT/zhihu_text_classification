# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:46:07 2018

@author: LIKS
"""

import sys,os
import pickle
import pandas as pd
from multiprocessing import Pool
import numpy as np

sys.path.append('../')
from data_helpers import pad_X30
from data_helpers import pad_X150
from data_helpers import pad_X52
from data_helpers import pad_X300
from data_helpers import train_batch
from data_helpers import eval_batch

wd_train_path = '../data/wd-data/data_train/'
wd_valid_path = '../data/wd-data/data_valid/'
wd_test_path = '../data/wd-data/data_test/'
ch_train_path = '../data/ch-data/data_train/'
ch_valid_path = '../data/ch-data/data_valid/'
ch_test_path = '../data/ch-data/data_test/'
paths = [wd_train_path, wd_valid_path, wd_test_path,
         ch_train_path, ch_valid_path, ch_test_path]

for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)
    

with open('../data/sr_topic2id.pkl','rb') as inp:
    sr_topic2id=pickle.load(inp)
dict_topic2id=dict()
for i in range(len(sr_topic2id)):
    dict_topic2id[sr_topic2id.index[i]]=sr_topic2id.values[i]

def topics2ids(topics):
    topics=topics.strip().split(',')
    #记得要将py3中的迭代器转化为List
    ids=list(map(lambda topic:dict_topic2id[topic],topics))
    return ids

def get_labels():
    #'注意删除前面丢弃的样本'
    df_question_topic=pd.read_csv('../raw_data/question_topic_train_set.txt',sep='\t',
                                  names=['questions','topics'],dtype={'questions':object,'topics':object})
    na_title_indexs = [328877, 422123, 633584, 768738, 818616, 876828, 1273673, 1527297,
                       1636237, 1682969, 2052477, 2628516, 2657464, 2904162, 2993517]
    df_question_topic=df_question_topic.drop(na_title_indexs)
    p=Pool()
    y=p.map(topics2ids,df_question_topic.topics.values)
    p.close()
    p.join()
    return np.asarray(y)

#数据打包
def wd_train_get_batch(title_len=30, content_len=150, batch_size=128):
    print('loading word train title and content...')
    train_title=np.load('../data/wd_train_title.npy')
    train_content=np.load('../data/wd_train_content.npy')
    p=Pool()
    title=np.asarray(p.map(pad_X30, train_title))
    content=np.asarray(p.map(pad_X150,train_content))
    p.close()
    p.join()
    X=np.hstack([title,content])
    print('getting labels, this should cost several minutes, please wait...')
    
    y=get_labels()
    print('y.shape=',y.shape)
    np.save('../data/y_tr.npy',y)
    
    sample_num=X.shape[0]
    np.random.seed(13)
    valid_num=10000
    new_index=np.random.permutation(sample_num)
    X=X[new_index]
    y=y[new_index]
    X_valid=X[:valid_num]
    y_valid=X[:valid_num]
    X_train=X[valid_num:]
    y_train=X[valid_num:]
    print('X_train.shape=',X_train.shape, 'y_train.shape=', y_train.shape)
    print('X_valid.shape=', X_valid.shape, 'y_valid.shape=', y_valid.shape)
    print('creating batch data.')
    #验证集打batch
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    train_batch(X_valid, y_valid, wd_valid_path, batch_size)
    # 训练集打batch
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    train_batch(X_train, y_train, wd_train_path, batch_size)

def wd_test_get_batch(title_len=30, content_len=150, batch_size=128):
    print('loading eval data...')
    title=np.load('../data/ch_eval_title.npy')
    content=np.load('../data/ch_eval_content.npy')
    print('transfer the eval data to fixed title_length 30 and content_length 150...')
    p=Pool()
    title=np.asarray(p.map(pad_X30,title))
    content=np.asarray(p.map(pad_X150,content))
    p.close()
    p.join()
    X=np.hstack([title,content])
    print('OK!transfered! titles and content are stacked horizontally')
    print('test sample_num=',len(X))
    eval_batch(X,wd_test_path,batch_size)
    
def ch_train_get_batch(title_len=52, content_len=300, batch_size=128):
    print('loading char train_title and train_content.')
    train_title = np.load('../data/ch_train_title.npy')
    train_content = np.load('../data/ch_train_content.npy')
    p = Pool()
    X_title = np.asarray(p.map(pad_X52, train_title))
    X_content = np.asarray(p.map(pad_X300, train_content))
    p.close()
    p.join()
    X = np.hstack([X_title, X_content])
    y = np.load('../data/y_tr.npy')
    # 划分验证集
    sample_num = X.shape[0]
    np.random.seed(13)
    valid_num = 100000
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = y[new_index]
    X_valid = X[:valid_num]
    y_valid = y[:valid_num]
    X_train = X[valid_num:]
    y_train = y[valid_num:]
    print('X_train.shape=', X_train.shape, 'y_train.shape=', y_train.shape)
    print('X_valid.shape=', X_valid.shape, 'y_valid.shape=', y_valid.shape)
    # 验证集打batch
    print('creating batch data.')
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    train_batch(X_valid, y_valid, ch_valid_path, batch_size)
    # 训练集打batch
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    train_batch(X_train, y_train, ch_train_path, batch_size)


def ch_test_get_batch(title_len=52, content_len=300, batch_size=128):
    eval_title = np.load('../data/ch_eval_title.npy')
    eval_content = np.load('../data/ch_eval_content.npy')
    p = Pool()
    X_title = np.asarray(p.map(pad_X52, eval_title))
    X_content = np.asarray(p.map(pad_X300, eval_content))
    p.close()
    p.join()
    X = np.hstack([X_title, X_content])
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, ch_test_path, batch_size)

if __name__ == '__main__':
    wd_train_get_batch()
    wd_test_get_batch()
    ch_train_get_batch()
    ch_test_get_batch()   
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    