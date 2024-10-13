#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:fangd123
@license: Apache Licence 
@file: test.py.py 
@time: 2020/11/24
@contact: fangd123@qq.com
@site:  
@software: PyCharm 
"""

import requests
import pandas as pd

def load_data(path:str):
    df = pd.read_excel(path,header=0,names=['label','text','a','b'])
    lines = df['text'].tolist()
    lines = [str(x) for x in lines]
    result = requests.post(url='http://192.168.1.17:9528/oneline/action',json={'texts':lines,'debug':True})
    data = result.json()
    predict = data['categories']
    prob = data['probabilities']
    prob = [x[i] for x,i in zip(prob,predict)]
    df['predict'] = predict
    df['prb'] = prob
    df.to_excel('预测后的.xlsx')
if __name__ == "__main__":
    load_data('data/raw/action_classify/动作场面标记数据1116(1).xlsx')