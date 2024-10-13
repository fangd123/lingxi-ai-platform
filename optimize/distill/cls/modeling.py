import logging
logger = logging.getLogger(__name__)

def BertForGLUESimpleAdaptor(batch, model_outputs, no_logits, no_mask):
    dict_obj = {'hidden': model_outputs[2], 'attention': model_outputs[3]}
    if no_mask is False:
        dict_obj['attention_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs[1],)
    return dict_obj

def BertForGLUESimpleAdaptorTraining(batch, model_outputs):
    return {'losses':(model_outputs[0],)}
