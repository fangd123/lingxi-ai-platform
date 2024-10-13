#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:fangd123
@license: Apache Licence 
@file: main.py 
@time: 2020/12/02
@contact: fangd123@qq.com
@site:  
@software: PyCharm 
"""
import torch
from loguru import logger

from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, default_data_collator
from onnxruntime.quantization import quantize_dynamic, QuantType

def load_model(config_path: str, model_path: str):
    bert_config = BertConfig.from_pretrained(config_path)
    bert_config.num_labels = 3
    model = BertForSequenceClassification(config=bert_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def export_onnx_model(model, onnx_model_path):
    with torch.no_grad():
        inputs = {'input_ids': torch.ones(1, 128, dtype=torch.int64),
                  'attention_mask': torch.ones(1, 128, dtype=torch.int64),
                  'token_type_ids': torch.ones(1, 128, dtype=torch.int64)}

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,  # model being run
                          (inputs['input_ids'],  # model input (or a tuple for multiple inputs)
                           inputs['attention_mask'],
                           inputs['token_type_ids']),  # model input (or a tuple for multiple inputs)
                          onnx_model_path,  # where to save the model (can be a file or file-like object)
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input_ids',  # the model's input names
                                       'input_mask',
                                       'segment_ids'],
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input_ids': symbolic_names,  # variable length axes
                                        'input_mask': symbolic_names,
                                        'segment_ids': symbolic_names,
                                        'output':symbolic_names})
        logger.info("ONNX Model exported to {0}".format(onnx_model_path))


def optimize(path='bert.onnx', out_path='bert.opt.onnx'):
    # disable embedding layer norm optimization for better model size reduction
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False

    opt_model = optimizer.optimize_model(
        path,
        'bert',
        num_heads=12,
        hidden_size=768,
        only_onnxruntime=True, # 若为false，则会报错
        optimization_options=opt_options)
    opt_model.save_model_to_file(out_path)


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    logger.info(f"quantized model saved to:{quantized_model_path}")


if __name__ == "__main__":
    model = load_model(config_path='/nfs/protech/模型库/预训练模型/script_novel_pretrain_bert_rbt3/config.json',
                       model_path='/home/hmqf/projects/line_temp/optimize/distill/output_root_dir/mnli_t8_TbaseST4tiny_AllSmmdH1_lr10e30_bs32/gs5928.pkl')
    export_onnx_model(model, "bert.onnx")
    optimize()
    quantize_onnx_model('bert.opt.onnx', 'bert.opt.quant.onnx')
