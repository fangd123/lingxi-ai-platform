import time

import numpy as np
from fastapi import APIRouter
from starlette.requests import Request
from fastapi_cache.decorator import cache

from api.config import settings
from api.schemas.Sentences import Sentences
from api.schemas.ClassifyResult import ClassifyResult
from api.utils import set_cache
from api.processor import convert_predict_examples_to_tensor_dataset
from api.utils import get_cached_results

router = APIRouter()


@router.post('/', response_model=ClassifyResult)
@cache(namespace=settings.NAMESPACE)
async def predict(sentences: Sentences,request:Request):
    texts = sentences.texts
    type = None
    debug = sentences.debug
    app = request.app
    # 处理空行
    idx_not_blank_texts = {i: text for i, text in enumerate(texts) if text.strip() != ''}
    not_blank_texts = [x for x in idx_not_blank_texts.values()]

    # 若全为空行，则直接输出结果
    if len(not_blank_texts) == 0:
        result = {"categories":[0] * len(texts)}
        return result

    # 记录处理的字数信息
    app.TEXT_LINE_COUNTER.labels(settings.PROJECT_NAME, 'POST', '/oneline').inc(len(texts))
    app.TEXT_LENGTH_COUNTER.labels(settings.PROJECT_NAME, 'POST', '/oneline').inc(sum([len(x) for x in texts]))

    cached_result, new_texts_id = await get_cached_results(app, texts)
    new_texts = [texts[i] for i in new_texts_id]
    if len(new_texts) > 0 and new_texts[0].strip() != '':
        start = time.time()
        dataloader = transform_text(app,new_texts, type)
        result = get_prediction(app,dataloader,debug)
        new_texts_result = [(i, x) for i, x in zip(new_texts_id, result[0])]
        app.logger.info(f'模型预测耗时:{time.time() - start}')
        cached_result.extend(new_texts_result)
        # 把分类结果添加入缓存当中
        await set_cache(app, new_texts, result[0])
    # 确保返回的数据个数和输入的一致
    assert len(cached_result) == len(texts)

    cached_result.sort(key=lambda x: x[0])
    if sentences.debug:
        result = {"categories": [x[1] for x in cached_result], "probabilities": result[1]}
    else:
        result = {"categories": [x[1] for x in cached_result]}
    app.logger.info(texts)
    app.logger.info(result)
    return result


def transform_text(app,texts: list, text_bs: list = None):
    if not text_bs:
        text_bs = [None] * len(texts)
    texts = [(i, a, b) for i, (a, b) in enumerate(zip(texts, text_bs))]
    examples = app.processor.create_examples(lines=texts, set_type='test')
    tensors = convert_predict_examples_to_tensor_dataset(
        examples,
        app.tokenizer,
        max_length=128
    )

    return tensors


def get_prediction(app,dataloader,debug):
    preds = []
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
    if debug:
        from scipy.special import softmax
        probs = softmax(preds,axis=-1)
        probs = probs.tolist()
    else:
        probs = None
    preds = np.argmax(preds, axis=1).tolist()
    return [preds, probs]
