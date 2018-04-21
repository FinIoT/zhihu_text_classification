# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:30:20 2018

@author: LIKS

模块功能：
-将所有字词的向量保存到.npy便于用numpy计算
-将所有字词Series，便于后面将训练数据数字化为整数1,2,3...的形式
-处理特殊字符如 PAD填充符 UNK罕见字符

该功能模块编程技巧不高，只需对各个类库熟悉
"""
import word2vec
import os
import pandas as pd
import numpy as np
import pickle

SPECIAL_SYMBOL=['<PAD>','<UNK>']
n_special_sym=len(SPECIAL_SYMBOL)
embedding_size=256


def get_word_embedding():
    print('start to load word2vec...')
    wv=word2vec.load('../raw_data/word_embedding.txt')
    word_embedding=wv.vectors
    words=wv.vocab
    
    #用Series存放word
    sr_id2word=pd.Series(words,index=range(n_special_sym,n_special_sym+len(words)))
    sr_word2id=pd.Series(range(n_special_sym,n_special_sym+len(words)),index=words)
    
    
    #Series中加入特殊字符
    for i in range(n_special_sym):
        sr_id2word[i]=SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]]=i
    
    #加入特殊字符的向量
    vec_special_sym=np.random.randn(n_special_sym,embedding_size)
    word_embedding=np.vstack((vec_special_sym,word_embedding))
     
    #保存词向量
    save_path='../data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+'word_embedding.npy',word_embedding)
    
    with open(save_path+'sr_word2id.pkl','wb') as outp:
        pickle.dump(sr_word2id,outp)
        pickle.dump(sr_id2word,outp)
    print('saving the word_embedding.npy to ../data/word_embedding,npy ')
    
    
#def get_char_embedding():

    
if __name__=='__main__':
    get_word_embedding()
#    get_char_embedding()