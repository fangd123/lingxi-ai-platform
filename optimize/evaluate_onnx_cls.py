# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2021/9/18 10:32 上午
@Author  : zhouhongzhe
@FileName: model_evaluate_result_cls.py
@Software: PyCharm
@Describe:
'''

# 评估优化后的ONNX模型
# 用于分类模型

import numpy as np
import pandas as pd
import onnxruntime as rt
from transformers import BertTokenizerFast
import sys

sys.path.append('..')
from deploy.api.processor import Processor
from deploy.api.processor import convert_predict_examples_to_tensor_dataset
import fire


def model_init(path):
    tokenizer = BertTokenizerFast.from_pretrained(path="../deploy/models", do_lower_case=True)
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession(path, sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                                 'CPUExecutionProvider'])
    processor = Processor()
    return session, tokenizer, processor


def transform_text(tokenizer, processor, texts: list, text_bs: list = None):
    if not text_bs:
        text_bs = [None] * len(texts)
    texts = [(i, a, b) for i, (a, b) in enumerate(zip(texts, text_bs))]
    examples = processor.create_examples(lines=texts, set_type='test')
    tensors = convert_predict_examples_to_tensor_dataset(
        examples,
        tokenizer,
        max_length=128
    )

    return tensors


def get_prediction(session,dataloader):
    preds = []
    for i in range(0, len(dataloader['input_ids']), 32):
        inputs = {'input_ids': dataloader['input_ids'][i:i + 32],
                  'attention_mask': dataloader['attention_mask'][i:i + 32],
                  'token_type_ids': dataloader['token_type_ids'][i:i + 32]}
        res = session.run(None, {
            'input_ids': inputs['input_ids'],
            'input_mask': inputs['attention_mask'],
            'segment_ids': inputs['token_type_ids']
        })
        logits = res[0]
        preds.append(logits)
    preds = np.concatenate(preds, axis=0)
    probs = None
    preds = np.argmax(preds, axis=1).tolist()
    return [preds, probs]


def predict(session, tokenizer, processor, texts: list):
    dataloader = transform_text(tokenizer, processor, texts)
    result = get_prediction(session, dataloader)
    result = {"categories": result[0]}
    return result


def main(test_file: str = '../data/test.csv', result_file='fp16模型onnx.csv', model_file=''):
    session, tokenizer, processor = model_init(model_file)
    with open(test_file, 'r', encoding='utf-8') as f:
        data = pd.read_table(f)
    data_dict = data.to_dict(orient='list')
    sentence = data_dict['sentence']
    result = predict(session, tokenizer, processor, sentence)
    data_real_label = data_dict['label']
    data_pred_label = result['categories']
    true_counter = 0
    index_list = [i for i in range(len(data_pred_label))]
    pre_dict = {}
    pre_dict['index'] = index_list
    pre_dict['prediction'] = data_pred_label
    with open(result_file, 'w', encoding='utf-8') as f:
        pre_df = pd.DataFrame(pre_dict)
        pre_df.to_csv(f, index=None, sep='\t', encoding='utf-8')
    for label_r, label_p in zip(data_real_label, data_pred_label):
        if label_p == label_r:
            true_counter += 1
    total_number = len(data_real_label)
    # 准确率
    accuracy_number = true_counter / total_number * 100
    print(accuracy_number)


if __name__ == "__main__":
    fire.Fire(main)
