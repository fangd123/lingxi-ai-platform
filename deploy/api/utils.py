import asyncio
import hashlib

import onnxruntime as rt
from transformers import BertTokenizerFast

from api.processor import Processor
from api.config import settings

def get_ip():
    import socket
    # 获取本机计算机名称
    hostname = socket.gethostname()
    # 获取本机ip
    ip = socket.gethostbyname(hostname)
    return ip


def model_init(path):
    tokenizer = BertTokenizerFast.from_pretrained(path=settings.MODEL_FOLDER_PATH, do_lower_case=True)
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession(path, sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    processor = Processor()
    return session, tokenizer, processor


def cache_key_builder(text):
    prefix = f"{settings.PROJECT_NAME}:{settings.NAMESPACE}:"
    cache_key = (prefix + hashlib.md5(text.encode()).hexdigest())
    return cache_key


async def get_cached_results(app, texts: list):
    """
    判断当前文本列表中是否有之前计算过的数据
    :param texts: 文本列表
    :return: 布尔列表，True 之前计算过，False 之前没算过
    """
    app.logger.info("获取redis缓存结果")
    cached_result = []
    new_texts_id = []
    results = await app.redis.mget([cache_key_builder(text) for text in texts])
    for i, result in enumerate(results):
        if result:
            cached_result.append((i, result))
        else:
            new_texts_id.append(i)
    app.logger.info(f"缓存结果{results}")
    return cached_result, new_texts_id


async def set_cache(app, texts: list, result: list):
    app.logger.info("将计算结果存入缓存中")
    cache_dict = {cache_key_builder(text): x for text, x in zip(texts, result)}
    asyncio.create_task(set_expire_cache(app.redis,cache_dict))

async def set_expire_cache(redis,cache_dict):
    await redis.mset(cache_dict)
    [await redis.expire(key,settings.CACHE_EXPIRE_TIME) for key in cache_dict.keys()]
