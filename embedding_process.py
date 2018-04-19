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
import pandas as pd

SPECIAL_SYMBOL=['PAD','UNK']
n_special_sym=len(SPECIAL_SYMBOL)


def get_word_embedding():
    print('start to load word2vec...')
    wv=word2vec.load('../raw_data/word_embedding.txt')
    word_embedding=wv.vectors
    words=wv.vocab
    
    #用Series存放word
    sr_id2word=pd.Series(words,index=range(n_special_sym,n_special_sym+len(words)))
    
def get_char_embedding():


if __name__=='__main__':
    get_word_embedding()
    get_char_embedding()