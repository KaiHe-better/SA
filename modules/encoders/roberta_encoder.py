import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel, RobertaTokenizer



class Roberta_encoder(nn.Module):

    def __init__(self, pretrain_path):
        nn.Module.__init__(self)
        self.my_model = RobertaModel.from_pretrained(pretrain_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
        self.tokenizer.bos_token = "[CLS]"
        self.tokenizer.eos_token = "[SEP]"
    
    def forward(self, inputs):
        logits = self.my_model(input_ids=inputs['input_ids'], 
                           attention_mask=inputs['attention_mask'], 
                           token_type_ids=inputs['token_type_ids'], 
                           output_hidden_states=True)
        
        return logits["last_hidden_state"]
