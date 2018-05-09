# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:00 2018

@author: LIKS
"""
import numpy as np
import pandas as pd
import time
import tqdm
import pickle
from multiprocessing import Pool


save_path='../data/'
with open(save_path+'sr_char2id.pkl','rb') as inp:
    #Series形式的数据
    sr_id2char=pickle.load(inp)
    sr_char2id=pickle.load(inp)
dict_char2id=dict()
for i in range(len(sr_char2id)):
    dict_char2id[sr_char2id.index[i]]=sr_char2id.value[i]

def get_char2id(chars):
    chars=chars.strip().split(',')
    ids=list

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
    title2num=np.asarray(p.map(get_char2id,df_eval.char_title.values))
    
        


    
    

if __name__=='__main__':
    test_char2id()
   