# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:00 2018

@author: LIKS
"""
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import pickle
from multiprocessing import Pool


save_path='../data/'
with open(save_path+'sr_char2id.pkl','rb') as inp:
    #Series形式的数据
    sr_id2char=pickle.load(inp)
    sr_char2id=pickle.load(inp)
dict_char2id=dict()
for i in range(len(sr_char2id)):
    dict_char2id[sr_char2id.index[i]]=sr_char2id.values[i]

def get_char2id(char):
    if char not in dict_char2id:
        return 1
    else:
        return dict_char2id[char]
    

def get_chars2id(chars):
    char=chars.strip().split(',')
    ids=list(map(get_char2id,char))
    return ids

def test_char2id():
    '''将测试集中所有字转化为数字'''
    time0=time.time()
    print('loading data...')
    df_eval=pd.read_csv('../raw_data/question_eval_set.txt',sep='\t',usecols=[0,1,3],
                        names=['question_id','char_title','char_content'],dtype={'question_id':object})
    #217630
    print('test question number:% d' %len(df_eval))
    #print(type(df_eval.char_title)):pandas.core.series.Series
    #print(type(df_eval)):pandas.core.frame.DataFrame
    #print(df_eval.char_tdf_eval.char_title.valuesitle.values[88]):打印的是行号为88的值，其实是第89个值
    #print(type(df_eval.char_title.values)):ndarray
    na_title_list=[]
    for i in range(len(df_eval)):
        char_title=df_eval.char_title.values[i]
        if type(char_title) is float:
            na_title_list.append(i)
    print('there are %d questions wihtout title.'%len(na_title_list))
    #下面两行代码逻辑估计有问题，如果没有title就一定有content吗？
    for i in na_title_list:
        df_eval.at[i,'char_title']=df_eval.at[i,'char_content']
        
    na_content_list=list()
    for i in tqdm(range(len(df_eval))):
        char_content=df_eval.char_content.values[i]
        if type(char_content) is float:
            na_content_list.append(i)
    print('there are %d questions wihtout title.'%len(na_content_list))
    
    for i in tqdm(na_content_list):
        df_eval.at[i,'char_content']=df_eval.at[i,'char_title']
    
    p=Pool()
    eval_title2num=np.asarray(p.map(get_chars2id,df_eval.char_title.values))
    np.save('../data/ch_eval_title.npy',eval_title2num)
    eval_content2num=np.asarray(p.map(get_chars2id,df_eval.char_content.values))
    np.save('../data/ch_eval_content.npy',eval_content2num)
    p.close()
    p.join()
    print('chars to numnbers complete. it takes %d seconds'%(time.time()-time0))
    
def train_char2id():
    '''将测试集中所有字转化为数字'''
    time0=time.time()
    print('loading data...')
    df_eval=pd.read_csv('../raw_data/question_train_set.txt',sep='\t',usecols=[0,1,3],
                        names=['question_id','char_title','char_content'],dtype={'question_id':object})
    #217630
    print('test question number:% d' %len(df_eval))
    #print(type(df_eval.char_title)):pandas.core.series.Series
    #print(type(df_eval)):pandas.core.frame.DataFrame
    #print(df_eval.char_tdf_eval.char_title.valuesitle.values[88]):打印的是行号为88的值，其实是第89个值
    #print(type(df_eval.char_title.values)):ndarray
    na_content_list=[]
    for i in tqdm(range(len(df_eval))):
        char_content=df_eval.char_content.values[i]
        if type(char_content) is float:
            na_content_list.append(i)
    print('there are %d questions wihtout title.'%len(na_content_list))
    #下面两行代码逻辑估计有问题，如果没有title就一定有content吗？
    for i in tqdm(na_content_list):
        df_eval.at[i,'char_content']=df_eval.at[i,'char_title']
        
    na_title_list=[328877, 422123, 633584, 768738, 818616, 876828, 1273673, 1527297,
              1636237, 1682969, 2052477, 2628516, 2657464, 2904162, 2993517]
    for i in range(len(df_eval)):
        char_title=df_eval.char_title.values[i]
        if type(char_title) is float:
            na_title_list.append(i)
    print('There are %d train questions without title.' % len(na_title_list))
    print('na_title:',na_title_list)
    df_eval=df_eval.drop(na_title_list)
    print('After dropping, training question number(should be 2999952) = %d' % len(df_eval))
    #转为 id 形式
    
   
    p=Pool()
    eval_title2num=np.asarray(p.map(get_chars2id,df_eval.char_title.values))
    np.save('../data/ch_train_title.npy',eval_title2num)
    eval_content2num=np.asarray(p.map(get_chars2id,df_eval.char_content.values))
    np.save('../data/ch_train_content.npy',eval_content2num)
    p.close()
    p.join()
    print('chars to numnbers complete. it takes %d seconds'%(time.time()-time0))
    
if __name__=='__main__':
    test_char2id()
    train_char2id()
   