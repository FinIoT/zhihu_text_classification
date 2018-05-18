# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:46:07 2018

@author: LIKS
"""

import sys
import pickle

sys.path.append('../')
from data_helpers import pad_X30

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