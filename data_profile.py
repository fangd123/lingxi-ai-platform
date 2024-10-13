#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:fangd123
@license: Apache Licence 
@file: data_profile.py 
@time: 2020/11/23
@contact: fangd123@qq.com
@site:  
@software: PyCharm 
"""
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

def data_profile(file_path:str,output_path:str)->None:
    df = pd.read_csv(file_path,sep='\t',index_col=0,header=0)
    profile = ProfileReport(df, title='Pandas Profiling Report')
    profile.to_file(output_path)

if __name__ == "__main__":
    data_profile('data/train.tsv', output_path='profile.html')