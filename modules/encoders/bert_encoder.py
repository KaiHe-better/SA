import torch
import torch.nn as nn
import numpy as np
# from transformers import BertTokenizer, BertModel
from torchmeta.modules import MetaModule
from utils.meta_transformers import BertTokenizer, BertModel


class BERT_encoder(MetaModule):

    def __init__(self, pretrain_path):
        super(BERT_encoder, self).__init__()
        self.my_model = BertModel.from_pretrained(pretrain_path, add_pooling_layer=False)
        # self.my_model = nn.Sequential(*list(my_model.children())[:-1])
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.tokenizer.bos_token = "[CLS]"
        self.tokenizer.eos_token = "[SEP]"
    
    def forward(self, inputs, params=None):
        logits = self.my_model(input_ids=inputs['input_ids'], 
                           attention_mask=inputs['attention_mask'], 
                           token_type_ids=inputs['token_type_ids'], 
                           output_hidden_states=True, params=self.get_subdict(params, 'my_model'))
        
        return logits["last_hidden_state"]
