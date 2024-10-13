#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:fangd123
@license: Apache Licence 
@file: distill.py 
@time: 2020/12/01
@contact: fangd123@qq.com
@site:  
@software: PyCharm 
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, List, TextIO

import transformers
from transformers.trainer_utils import is_main_process, set_seed

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")
import sys
import argparse

sys.path.append('../..')
import os, random
import numpy as np
import torch

from transformers import BertConfig, BertTokenizerFast, DataCollatorForTokenClassification
from transformers import BertForTokenClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils import divide_parameters
from modeling import BertForGLUESimpleAdaptor
from matches import matches
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from functools import partial
from datasets import Dataset
import inspect
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


class NER:
    def __init__(self, label_idx=-1):
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx

    def read_examples_from_file(self, file_path) -> List[InputExample]:
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split("\t")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[self.label_idx].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=f"{guid_index}", words=words, labels=labels))
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu


def remove_unused_columns(model, dataset: "datasets.Dataset", description: Optional[str] = None):
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


def parse(opt=None):
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--num_labels", default=2, type=int, required=True)
    parser.add_argument("--task_name", default='wnli', type=str)
    parser.add_argument("--aux_task_name", default=None, type=str)
    ## Other parameters
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precisoin instead of 32-bit")

    parser.add_argument('--random_seed', type=int, default=10236797)
    parser.add_argument('--load_model_type', type=str, default='bert', choices=['bert', 'all', 'none'])
    parser.add_argument('--weight_decay_rate', type=float, default=0.01)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--PRINT_EVERY', type=int, default=200)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--ckpt_frequency', type=int, default=2)

    parser.add_argument('--tuned_checkpoint_T', type=str, default=None)
    parser.add_argument('--tuned_checkpoint_Ts', nargs='*', type=str)

    parser.add_argument('--tuned_checkpoint_S', type=str, default=None)
    parser.add_argument("--init_checkpoint_S", default=None, type=str)
    parser.add_argument("--bert_config_file_T", default=None, type=str, required=True)
    parser.add_argument("--bert_config_file_S", default=None, type=str, required=True)
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--teacher_cached", action='store_true')

    parser.add_argument('--s_opt1', type=float, default=1.0, help="release_start / step1 / ratio")
    parser.add_argument('--s_opt2', type=float, default=0.0, help="release_level / step2")
    parser.add_argument('--s_opt3', type=float, default=1.0, help="not used / decay rate")
    parser.add_argument('--schedule', type=str, default='warmup_linear_release')

    parser.add_argument('--no_inputs_mask', action='store_true')
    parser.add_argument('--no_logits', action='store_true')
    parser.add_argument('--output_att_score', default='true', choices=['true', 'false'])
    parser.add_argument('--output_att_sum', default='false', choices=['true', 'false'])
    parser.add_argument('--output_encoded_layers', default='true', choices=['true', 'false'])
    parser.add_argument('--output_attention_layers', default='true', choices=['true', 'false'])
    parser.add_argument('--matches', nargs='*', type=str)

    parser.add_argument('--only_load_embedding', action='store_true')
    if opt is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)
    return args


def main():
    # parse arguments
    args = parse()
    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")
    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # arguments check
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    # load bert config
    bert_config_T = BertConfig.from_json_file(args.bert_config_file_T)
    bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_T.max_position_embeddings
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    num_labels = args.num_labels
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", args)

    # Set seed before initializing model.
    set_seed(args.random_seed)
    token_classification_task = NER()

    data_files = {}
    data_files["train"] = args.data_dir+'/train.txt'
    data_files["validation"] = args.data_dir+'/dev.txt'
    data_files["test"] = args.data_dir+'/test.txt'

    datasets = {}
    for mode, path in data_files.items():
        examples = token_classification_task.read_examples_from_file(path)
        example_dict_by_column = defaultdict(list)
        first = examples[0]
        column_names = list(asdict(first).keys())
        for example in examples:
            example = asdict(example)
            for name in column_names:
                example_dict_by_column[name].append(example[name])
        dataset = Dataset.from_dict(dict(example_dict_by_column))
        datasets[mode] = dataset

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    if args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "words" if "words" in column_names else column_names[1]
    label_column_name = (
        f"labels" if f"labels" in column_names else column_names[-1]
    )

    label_list = token_classification_task.get_labels(args.data_dir+'/labels.txt')
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Preprocessing the dataset
    # Padding strategy
    padding = "longest"

    tokenizer = BertTokenizerFast(vocab_file=args.vocab_file)

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = {}
    for mode, dataset in datasets.items():
        tokenized_datasets[mode] = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=5,
        )
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    # Build Model and load checkpoint
    bert_config_T.num_labels = num_labels
    bert_config_S.num_labels = num_labels
    bert_config_T.output_attentions = args.output_attention_layers
    bert_config_T.output_hidden_states = args.output_encoded_layers
    bert_config_S.output_attentions = args.output_attention_layers
    bert_config_S.output_hidden_states = args.output_encoded_layers
    model_T = BertForTokenClassification(bert_config_T)
    model_S = BertForTokenClassification(bert_config_S)
    # Load teacher
    if args.tuned_checkpoint_T is not None:
        state_dict_T = torch.load(args.tuned_checkpoint_T, map_location='cpu')
        model_T.load_state_dict(state_dict_T)
        model_T.eval()
    else:
        assert args.do_predict is True
    # Load student
    if args.load_model_type == 'bert':
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        if args.only_load_embedding:
            state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.embeddings')}
            missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
            logger.info(f"Missing keys {list(missing_keys)}")
        else:
            state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.')}
            missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
            # print(missing_keys)
            # assert len(missing_keys) == 0
        logger.info("Model loaded")
    elif args.load_model_type == 'all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S, map_location='cpu')
        model_S.load_state_dict(state_dict_S)
        logger.info("Model loaded")
    else:
        logger.info("Model is randomly initialized.")
    model_T.to(device)
    model_S.to(device)

    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            model_T = torch.nn.DataParallel(model_T)  # ,output_device=n_gpu-1)
            model_S = torch.nn.DataParallel(model_S)  # ,output_device=n_gpu-1)

    eval_datasets = tokenized_datasets["validation"]

    def predict(model, eval_dataset, data_collator, step, args):
        eval_output_dir = args.output_dir
        results = {}
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
            eval_dataset)
        remove_unused_columns(model, eval_dataset, description="evaluation")
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.predict_batch_size,
            collate_fn=data_collator,
        )
        model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            labels = batch['labels']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']

            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            cpu_logits = logits[0].detach().cpu()
            for i in range(len(cpu_logits)):
                pred_logits.append(cpu_logits[i].numpy())
                label_ids.append(labels[i])

        pred_logits = np.array(pred_logits)
        label_ids = np.array(label_ids)
        result = compute_metrics((pred_logits, labels))
        logger.info(f"result: {result}")
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} task {} *****".format(step, ''))
            writer.write("step: %d ****\n " % step)
            for key in sorted(results.keys()):
                logger.info("%s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
        model.train()
        return results

    if args.do_train:
        # parameters
        train_dataset = tokenized_datasets["train"]
        num_train_steps = int(len(train_dataset) / args.train_batch_size) * args.num_train_epochs
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(args, params, lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))
        optimizer = AdamW(all_trainable_params, lr=args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * args.warmup_proportion),
                                                    num_training_steps=num_train_steps)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)

        ########### DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ckpt_frequency=args.ckpt_frequency,
            log_dir=args.output_dir,
            output_dir=args.output_dir,
            device=args.device)

        intermediate_matches = None
        if isinstance(args.matches, (list, tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        logger.info(f"{intermediate_matches}")
        distill_config = DistillationConfig(
            temperature=args.temperature,
            intermediate_matches=intermediate_matches)

        logger.info(f"{train_config}")
        logger.info(f"{distill_config}")
        adaptor_T = partial(BertForGLUESimpleAdaptor, no_logits=args.no_logits, no_mask=args.no_inputs_mask)
        adaptor_S = partial(BertForGLUESimpleAdaptor, no_logits=args.no_logits, no_mask=args.no_inputs_mask)

        distiller = GeneralDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=model_T, model_S=model_S,
                                     adaptor_T=adaptor_T,
                                     adaptor_S=adaptor_S)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError

        remove_unused_columns(model_S,train_dataset, description="training")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.forward_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True,
        )

        callback_func = partial(predict, eval_dataset=eval_datasets, data_collator=data_collator, args=args)
        with distiller:
            distiller.train(optimizer, scheduler=scheduler, dataloader=train_dataloader,
                            num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S, eval_datasets, data_collator=data_collator,step=0, args=args)
        print(res)


if __name__ == "__main__":
    main()
