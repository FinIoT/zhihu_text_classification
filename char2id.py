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
                        names=['question_id','char_title','char_content'],dtype={'question':object})
    print('test question number:% d' %len(df_eval))