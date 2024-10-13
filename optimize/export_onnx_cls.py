#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:fangd123
@license: Apache Licence
@file: 导出ONNX.py
@time: 2020/02/27
@contact: fangd123@qq.com
@site:
@software: PyCharm
"""
import torch
from transformers import BertConfig, BertForSequenceClassification
import os
import tempfile
import onnx
from pathlib import Path
import fire

def export_ONNX(pytorch_model='', onnx_model=''):
    model = BertForSequenceClassification.from_pretrained(pytorch_model)

    model.eval()
    inputs = {
        'input_ids': torch.ones((1, 512), dtype=torch.long),
        'attention_mask': torch.ones((1, 512), dtype=torch.long),
        'token_type_ids': torch.ones((1, 512), dtype=torch.long)
    }
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,  # model being run
                          (inputs['input_ids'],  # model input (or a tuple for multiple inputs)
                           inputs['attention_mask'],
                           inputs['token_type_ids']),
                          onnx_model,  # where to save the model (can be a file or file-like object)
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input_ids',  # the model's input names
                                       'input_mask',
                                       'segment_ids'],
                          output_names=['bios'],  # the model's output names
                          dynamic_axes={'input_ids': symbolic_names,  # variable length axes
                                        'input_mask': symbolic_names,
                                        'segment_ids': symbolic_names,
                                        'bios': [0]
                                        })


if __name__ == "__main__":
    fire.Fire(export_ONNX)