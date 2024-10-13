import inspect
from typing import Optional

from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, default_data_collator
from onnxruntime.quantization import quantize_dynamic, QuantType
from datasets import load_metric, load_dataset
from transformers import BertTokenizerFast
import numpy as np
import onnxruntime as rt
from loguru import logger

padding = "longest"
max_length = 128
predict_batch_size = 128
vocab_file = '../../output_dir/vocab.txt'

def remove_unused_columns(model,dataset: "datasets.Dataset", description: Optional[str] = None):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dset_description = "" if description is None else f"in the {description} set "
    logger.info(
        f"The following columns {dset_description}don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
    )
    dataset.set_format(type=dataset.format["type"], columns=columns)




def eval_cls(model_path:str='bert.opt.quant.onnx',data_dir: str = '../../data'):
    """
    评估优化后以及量化后的模型准确率
    :return:
    """
    metric = load_metric('glue', 'sst2')

    datasets = load_dataset("csv", data_files={"validation": data_dir + '/eval.csv'}, delimiter='\t')
    non_label_column_names = [name for name in datasets["validation"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 8:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[1], None

    tokenizer = BertTokenizerFast(vocab_file=vocab_file)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    eval_dataset = datasets["validation"]
    eval_sampler = SequentialSampler(eval_dataset)

    raw_model = BertForSequenceClassification(config=BertConfig())
    remove_unused_columns(raw_model, eval_dataset, description="evaluation")

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=predict_batch_size,
        collate_fn=default_data_collator,
    )

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession('bert.opt.quant.onnx', sess_options)

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
        labels = batch['labels']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        inputs = {'input_ids': input_ids.cpu().numpy(),
                  'attention_mask': attention_mask.cpu().numpy(),
                  'token_type_ids': token_type_ids.cpu().numpy()}

        res = session.run(None, {
            'input_ids': inputs['input_ids'],
            'input_mask': inputs['attention_mask'],
            'segment_ids': inputs['token_type_ids']
        })

        preds = res[0]
        preds = np.argmax(preds, axis=1).tolist()
        metric.add_batch(predictions=preds, references=labels)

    score = metric.compute()
    print(score)


    
if __name__ == '__main__':
    eval_cls()