# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:33:55 2018

@author: LIKS
"""
import pandas as pd
def question_and_topic_2id():
    df_question_topic=pd.read_csv('../raw_data/question_topic_train_set.txt',sep='\t',names=['question','topics']
                                  ,dtype={'question':object,'topics':object})
    #print(type(df_question_topic.topics))
    #df_question_topic.topics.values为ndarray,数据样例['7739004195693774975,3738968195649774859' '-3149765934180654494'...
    #'-6440461292041887516,-7283993654004755131,-5378699121209676383']
    
    df_question_topic.topics=df_question_topic.topics.apply(lambda tps:tps.split(','))
    save_path='../data'
    print('question number:%d' %len(df_question_topic))
    
    questions=df_question_topic.question.values
   
    sr_question2id=pd.Series(range(len(questions)),index=questions)
    sr_id2question=pd.Series(questions,index=range(len(questions)))


if __name__=='__main__':
    question_and_topic_2id()