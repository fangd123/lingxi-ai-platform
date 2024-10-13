import json
import time
from collections import defaultdict

import numpy as np
from fastapi import APIRouter
from starlette.requests import Request
from fastapi_cache.decorator import cache

from api.config import settings
from api.schemas.Sentences import Sentences
from api.schemas.NerResult import NerResult
from api.utils import set_cache
from api.processor import convert_predict_examples_to_tensor_dataset
from api.utils import get_cached_results

router = APIRouter()

label_list = ['O', 'B-说话人', 'I-说话人  ']
num_labels = len(label_list)
id2label = {i: x for i, x in enumerate(label_list, 0)}


@router.post('/novel/shr', response_model=NerResult)
@cache(namespace=settings.NAMESPACE)
async def predict(sentences: Sentences,request:Request):
    """
    此版本不考虑空行需要一一对应的问题，直接忽略空行
    :param sentences:
    :param request:
    :return:
    """
    texts = sentences.texts
    type = sentences.type
    app = request.app
    # 忽略空行
    texts = [x for x in texts if x.strip() != '']

    # 记录处理的字数信息
    app.TEXT_LINE_COUNTER.labels(settings.PROJECT_NAME, 'POST', '/oneline').inc(len(texts))
    app.TEXT_LENGTH_COUNTER.labels(settings.PROJECT_NAME, 'POST', '/oneline').inc(sum([len(x) for x in texts]))

    cached_result, new_texts_id = await get_cached_results(app, texts)
    cached_result = [(i,json.loads(x)) for i,x in cached_result]
    new_texts = [texts[i] for i in new_texts_id]
    if len(new_texts) > 0 and new_texts[0].strip() != '':
        start = time.time()
        dataloader,sample_mapping, offset_mapping = transform_text(app,new_texts, type)
        result = get_prediction(app,dataloader)
        new_finds = result_parse_role(new_texts, result, sample_mapping, offset_mapping)
        app.logger.info(f'模型预测耗时:{time.time() - start}')
        # 把分类结果添加入缓存当中
        one_text_finds = convert_per_line(new_texts,new_finds)
        await set_cache(app, new_texts, [json.dumps(x) for x in one_text_finds])

        new_texts_result = [(i, x) for i, x in zip(new_texts_id, one_text_finds)]
        cached_result.extend(new_texts_result)
    # 确保返回的数据个数和输入的一致
    assert len(cached_result) == len(texts)
    cached_result.sort(key=lambda x: x[0])
    BIO_result = convert_whole_line([x[1] for x in cached_result])
    app.logger.info(BIO_result)
    return BIO_result

def convert_per_line(texts:list,finds:tuple):
    """
    将返回的计算格式转换为每句话的形式
    :param texts: 原文本
    :param finds: 实体
    :return:
    """
    all_finds = [{} for x in range(len(texts))]
    for entity_name,all_entity in finds.items():
        for i,entity in enumerate(all_entity):
            all_finds[i][entity_name]=entity
    return all_finds


def convert_whole_line(finds:list):
    """
    将返回的计算格式转换为整体一个好多句话的格式
    :param texts: 原文本
    :param finds: 实体
    :return:
    """
    all_finds = defaultdict(list)
    entity_name = finds[0].keys()
    for entity in finds:
        for one_name in entity_name:
            all_finds[one_name].append(entity[one_name])
    return all_finds

def transform_text(app,texts: list, text_bs: list = None):
    if not text_bs:
        text_bs = [None] * len(texts)
    texts = [(i, a, b) for i, (a, b) in enumerate(zip(texts, text_bs))]
    examples = app.processor.create_examples(lines=texts, set_type='test')
    tensors,sample_mapping, offset_mapping = convert_predict_examples_to_tensor_dataset(
        examples,
        app.tokenizer,
        max_length=128,
        task='ner'
    )

    return tensors,sample_mapping, offset_mapping


def get_prediction(app,dataloader):
    preds = []
    preds_BIO = []
    for i in range(0,len(dataloader['input_ids']),32):
        inputs = {'input_ids': dataloader['input_ids'][i:i+32],
                  'attention_mask': dataloader['attention_mask'][i:i+32],
                  'token_type_ids': dataloader['token_type_ids'][i:i+32]}
        res = app.dl_model.run(None, {
            'input_ids': inputs['input_ids'],
            'input_mask': inputs['attention_mask'],
            'segment_ids': inputs['token_type_ids']
        })
        logits = res[0]
        preds.append(logits)
    preds = np.concatenate(preds, axis=0)
    preds_list = np.argmax(preds, axis=2).tolist()
    for line in preds_list:
        preds_BIO.append([id2label[int(x)] for x in line[1:129]])
    return preds_BIO

def result_parse_role(dialog_text, raw_result, sample_mapping='', offset_mapping=''):
    """
    解析预测的结果
    包含role标签
    """
    import re
    b_labels = ['B-说话人', 'B-删除', 'B-新行']
    i_labels = ['I-说话人', 'I-删除', 'I-新行']
    all_speaker = defaultdict(list)
    all_delete = defaultdict(list)
    all_newline = defaultdict(list)
    for i, offset in enumerate(offset_mapping):
        text_i = sample_mapping[i]  # 句子级别的序号，当一句话别分割成了多句话，例如[0,0,1,1]，两句话被分割成了四句
        full_text = dialog_text[text_i]  # 找到未分割的原句
        true_text = []
        # 当前分割后的句子内容
        for s, j in offset:
            if full_text[s // 2:(j + 1) // 2] == '':
                continue
            true_text.append(full_text[s // 2:(j + 1) // 2])
        true_text = ''.join(true_text)
        # 当前分割后的句子的模型预测结果
        ners = raw_result[i]

        # 实体匹配
        flags = ' '.join(ners)
        for i_label in range(len(b_labels)):
            if i_label == 0:
                flags = flags.replace(b_labels[i_label], 's')
                flags = flags.replace(i_labels[i_label], 't')
            elif i_label == 1:
                flags = flags.replace(b_labels[i_label], 'r')
                flags = flags.replace(i_labels[i_label], 'n')
            elif i_label == 2:
                flags = flags.replace(b_labels[i_label], 'g')
                flags = flags.replace(i_labels[i_label], 'j')

        flags = ''.join(flags.split())
        result_speaker = re.finditer('st*', flags)
        speakers = ner_json_output(result_speaker, true_text, offset)
        all_speaker[text_i].extend(speakers)
        result_delete = re.finditer('rn*', flags)
        deletes = ner_json_output(result_delete, true_text, offset)
        all_delete[text_i].extend(deletes)
        result_newline = re.finditer('gj*', flags)
        newlines = ner_json_output(result_newline, true_text, offset)
        all_newline[text_i].extend(newlines)

    all_speakers = [all_speaker[k] for k in sorted(all_speaker.keys())]
    all_deletes = [all_delete[k] for k in sorted(all_delete.keys())]
    all_newlines = [all_newline[k] for k in sorted(all_newline.keys())]
    #return {'speakers': all_speakers, 'deletes': all_deletes, 'newlines': all_newlines}
    return {'speakers': all_speakers}


def ner_json_output(result, true_text, offset):
    result_list = []
    for one in result:
        if true_text[one.start():one.end()] == '':
            continue
        result_list.append({'start': (offset[one.start()+1][0]+1)//2,
                             'end': (offset[one.end()][1]+1)//2,
                             'text': true_text[one.start():one.end()]})
    return result_list