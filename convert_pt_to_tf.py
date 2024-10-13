# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Convert pytorch checkpoints to TensorFlow"""


import argparse
import os

from transformers import (
    ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
    DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
    DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    WEIGHTS_NAME,
    XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AlbertConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    FlaubertConfig,
    GPT2Config,
    LayoutLMConfig,
    LxmertConfig,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TFAlbertForPreTraining,
    TFBartForConditionalGeneration,
    TFBertForPreTraining,
    TFBertForMaskedLM,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFCamembertForMaskedLM,
    TFCTRLLMHeadModel,
    TFDistilBertForMaskedLM,
    TFDistilBertForQuestionAnswering,
    TFDPRContextEncoder,
    TFDPRQuestionEncoder,
    TFDPRReader,
    TFElectraForPreTraining,
    TFFlaubertWithLMHeadModel,
    TFGPT2LMHeadModel,
    TFLayoutLMForMaskedLM,
    TFLxmertForPreTraining,
    TFLxmertVisualFeatureEncoder,
    TFOpenAIGPTLMHeadModel,
    TFRobertaForCausalLM,
    TFRobertaForMaskedLM,
    TFRobertaForSequenceClassification,
    TFT5ForConditionalGeneration,
    TFTransfoXLLMHeadModel,
    TFWav2Vec2Model,
    TFXLMRobertaForMaskedLM,
    TFXLMWithLMHeadModel,
    TFXLNetLMHeadModel,
    TransfoXLConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    cached_path,
    is_torch_available,
    load_pytorch_checkpoint_in_tf2_model,
)
from transformers.utils import logging


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        AlbertForPreTraining,
        BartForConditionalGeneration,
        BertForPreTraining,
        BertForMaskedLM,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        CamembertForMaskedLM,
        CTRLLMHeadModel,
        DistilBertForMaskedLM,
        DistilBertForQuestionAnswering,
        DPRContextEncoder,
        DPRQuestionEncoder,
        DPRReader,
        ElectraForPreTraining,
        FlaubertWithLMHeadModel,
        GPT2LMHeadModel,
        LayoutLMForMaskedLM,
        LxmertForPreTraining,
        LxmertVisualFeatureEncoder,
        OpenAIGPTLMHeadModel,
        RobertaForMaskedLM,
        RobertaForSequenceClassification,
        T5ForConditionalGeneration,
        TransfoXLLMHeadModel,
        XLMRobertaForMaskedLM,
        XLMWithLMHeadModel,
        XLNetLMHeadModel,
    )


logging.set_verbosity_info()

MODEL_CLASSES = {
    "bart": (
        BartConfig,
        TFBartForConditionalGeneration,
        BartForConditionalGeneration,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "bert": (
        BertConfig,
        TFBertForMaskedLM,
        BertForMaskedLM,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-large-cased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-base-cased-finetuned-mrpc": (
        BertConfig,
        TFBertForSequenceClassification,
        BertForSequenceClassification,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "dpr": (
        DPRConfig,
        TFDPRQuestionEncoder,
        TFDPRContextEncoder,
        TFDPRReader,
        DPRQuestionEncoder,
        DPRContextEncoder,
        DPRReader,
        DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "gpt2": (
        GPT2Config,
        TFGPT2LMHeadModel,
        GPT2LMHeadModel,
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlnet": (
        XLNetConfig,
        TFXLNetLMHeadModel,
        XLNetLMHeadModel,
        XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlm": (
        XLMConfig,
        TFXLMWithLMHeadModel,
        XLMWithLMHeadModel,
        XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlm-roberta": (
        XLMRobertaConfig,
        TFXLMRobertaForMaskedLM,
        XLMRobertaForMaskedLM,
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "transfo-xl": (
        TransfoXLConfig,
        TFTransfoXLLMHeadModel,
        TransfoXLLMHeadModel,
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "openai-gpt": (
        OpenAIGPTConfig,
        TFOpenAIGPTLMHeadModel,
        OpenAIGPTLMHeadModel,
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "roberta": (
        RobertaConfig,
        TFRobertaForCausalLM,
        TFRobertaForMaskedLM,
        RobertaForMaskedLM,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "layoutlm": (
        LayoutLMConfig,
        TFLayoutLMForMaskedLM,
        LayoutLMForMaskedLM,
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "roberta-large-mnli": (
        RobertaConfig,
        TFRobertaForSequenceClassification,
        RobertaForSequenceClassification,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "camembert": (
        CamembertConfig,
        TFCamembertForMaskedLM,
        CamembertForMaskedLM,
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "flaubert": (
        FlaubertConfig,
        TFFlaubertWithLMHeadModel,
        FlaubertWithLMHeadModel,
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "distilbert": (
        DistilBertConfig,
        TFDistilBertForMaskedLM,
        DistilBertForMaskedLM,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "distilbert-base-distilled-squad": (
        DistilBertConfig,
        TFDistilBertForQuestionAnswering,
        DistilBertForQuestionAnswering,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "lxmert": (
        LxmertConfig,
        TFLxmertForPreTraining,
        LxmertForPreTraining,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "lxmert-visual-feature-encoder": (
        LxmertConfig,
        TFLxmertVisualFeatureEncoder,
        LxmertVisualFeatureEncoder,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "ctrl": (
        CTRLConfig,
        TFCTRLLMHeadModel,
        CTRLLMHeadModel,
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "albert": (
        AlbertConfig,
        TFAlbertForPreTraining,
        AlbertForPreTraining,
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "t5": (
        T5Config,
        TFT5ForConditionalGeneration,
        T5ForConditionalGeneration,
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "electra": (
        ElectraConfig,
        TFElectraForPreTraining,
        ElectraForPreTraining,
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "wav2vec2": (
        Wav2Vec2Config,
        TFWav2Vec2Model,
        Wav2Vec2Model,
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
}


def convert_pt_checkpoint_to_tf(
    model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False):
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unrecognized model type, should be one of {list(MODEL_CLASSES.keys())}.")

    config_class, model_class, pt_model_class, aws_config_map = MODEL_CLASSES['bert']

    # Initialise TF model
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print(f"Building TensorFlow model from configuration: {config}")
    tf_model = model_class(config)

    # Load PyTorch checkpoint in tf2 model:
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    if compare_with_pt_model:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # build the network

        state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
        pt_model = pt_model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print(f"Max absolute difference between models outputs {diff}")
        assert diff <= 2e-2, f"Error, model absolute difference is >2e-2: {diff}"

    # Save pytorch-model
    print(f"Save TensorFlow model to {tf_dump_path}")
    tf_model.save_weights(tf_dump_path, save_format="h5")


if __name__ == "__main__":
    convert_pt_checkpoint_to_tf('bert','bert_base_script_fintune/pytorch_model.bin','bert_base_script_fintune/config.json','abc.h5')