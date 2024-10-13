#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:fangd123
@license: Apache Licence 
@file: create_database.py 
@time: 2020/11/18
@contact: fangd123@qq.com
@site:  
@software: PyCharm 
"""
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl
from pathlib import Path


def tokenizer(sentence: str) -> list:
    return [x for x in sentence]


# files = Path('data/raw_data').glob('*.txt')
files = Path('data/第二次标注数据/project1').glob('*.txt')

sentences = []
for file in files:
    dataset = read_jsonl(filepath=str(file), dataset=NERDataset, encoding='utf-8')
    conll = dataset.to_conll2003(tokenizer=tokenizer)
    sentences.extend([x['data'].replace(' _ _ ', '\t') for x in conll])

## 切分数据库
# dir = "data/"
dir = "../data/第二次标注数据/project1_tsv"
import random

random.shuffle(sentences)

with open(dir + 'train.tsv', 'w', encoding='utf-8') as train, \
        open(dir + 'dev.tsv', 'w') as dev, \
        open(dir + 'test.tsv', 'w') as test:
    train.writelines(sentences[:3500])
    dev.writelines(sentences[3500:4200])
    test.writelines(sentences[4200:])
