# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:00 2018

@author: LIKS
"""
import numpy as np
import pandas as pd
import time

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
    nan_list=[]
    for i in enumerate(df_eval.char_content.isna()):
        if i[1]==True:
            nan_list.append(i[0])
    #print('non content number: %d'% (len(nan_list))):55179     
#    for i in nan_list:
#        print(i)

    
    

if __name__=='__main__':
    test_char2id()
   