#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:fangd123
@license: Apache Licence
@file: processor.py
@time: 2020/11/30
@contact: fangd123@qq.com
@site:
@software: PyCharm
"""
import csv
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def convert_predict_examples_to_tensor_dataset(
        examples: List[InputExample],
        tokenizer,
        max_length: Optional[int] = 128,
        task: str = ''
):
    if max_length is None:
        max_length = tokenizer.max_len
    labels = [None] * len(examples)

    if task == 'ner':
        texts = [example.text_a for example in examples]
        texts = [' '.join([x for x in one]) for one in texts]
        # 这里需要考虑超长文本对于NER识别的影响
        # 这里采用的是重叠处理的方式
        # 即按照最大长度把文本分为多个部分，每一部分之间会包含重叠的文本，重叠的长度由stride控制
        # 具体的参数解释可以参考https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
        batch_encoding = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            stride=10,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True)

        sample_mapping = batch_encoding.pop("overflow_to_sample_mapping")
        offset_mapping = batch_encoding.pop("offset_mapping")

        return batch_encoding, sample_mapping, offset_mapping

    elif task == 'ner_short':
        texts = [example.text_a for example in examples]
        texts = [' '.join([x for x in one]) for one in texts]
        batch_encoding = tokenizer(texts,
                                   max_length=max_length,
                                   padding=True,
                                   truncation=True)

        return batch_encoding

    else:
        if examples[0].text_b:
            pair = [(example.text_a, example.text_b) for example in examples]
        else:
            pair = [example.text_a for example in examples]

        batch_encoding = tokenizer(pair,
                                   max_length=max_length,
                                   padding=True,
                                   truncation=True)




    return batch_encoding


class Processor():
    """Processor for the WNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def create_examples(self, lines, set_type, task=''):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                if 'index' in str(line[0]):
                    continue
            guid = "%s-%s" % (set_type, line[0])
            if task == 'ner':
                text_a = [x for x in line[1]]
            else:
                text_a = line[1]
            if len(line) == 3:
                text_b = None
            else:
                text_b = line[2]
            label = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
