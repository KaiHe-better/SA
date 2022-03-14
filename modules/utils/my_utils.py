import time
import torch
import torch.nn as nn
import os
import sys
from collections import OrderedDict
from torchmeta.modules import MetaModule
import numpy as np
import torch.nn.functional as F

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



class Print_and_write:
    def __init__(self, file_name):
        self.f = file_name
        
    def write(self, text):
        print(text, end="")
        my_f = open(self.f, 'a', encoding='utf-8')
        my_f.write(text)
        my_f.close()
        
    # def close(self):
    #     self.f.close()
        
def print_execute_time(func):
    if type(func) is tuple:
        opera_time = (func[1] - func[0])
        if opera_time > 60:
            if opera_time / 60 > 60:
                opera_time = round(opera_time / 3600, 3)
                print(f'execute time: {opera_time} hour')
            else:
                opera_time = round(opera_time / 60, 3)
                print(f'execute time: {opera_time} minute')
        else:
            print(f'execute time: {round(opera_time, 3)} s')
    else:
        def wrapper(*args, **kwargs):
            start = time.time()
            func_return = func(*args, **kwargs)
            end = time.time()
            opera_time = (end - start)
            if opera_time > 60:
                if opera_time / 60 > 60:
                    opera_time = round(opera_time / 3600, 3)
                    print(f'execute time: {opera_time} hour')
                else:
                    opera_time = round(opera_time / 60, 3)
                    print(f'execute time: {opera_time} minute')
            else:
                print(f'execute time: {round(opera_time, 3)} s')

            return func_return

        return wrapper

def data_to_device(tensor_data_list):
    # index_list, bert_input_per, bert_mask_per, bert_segment_per, ae_labels, oe_labels, sc_labels
    my_input ={ 'input_ids': tensor_data_list[1],
                'attention_mask': tensor_data_list[2],
                # 'sentiment_mask': tensor_data_list[3], 
                'token_type_ids': tensor_data_list[3] }
        
    my_target ={'target_loss_labels': tensor_data_list[4],
                'opinion_loss_labels': tensor_data_list[5], 
                'polarity_loss_labels': tensor_data_list[6], 
                'index': tensor_data_list[0],
                }
    
    if torch.cuda.is_available():
        for i, v in my_input.items():
            my_input[i] = v.cuda()
        
        for i, v in my_target.items():
            if i != "index" and \
                i!="tokens" and \
                i!="tag_to_index" and  \
                i!=  "aspect_to_tag_dic"  and  \
                i!=  "opinion_to_tag_dic" and  \
                i!=  "sentiment_to_tag_dic":
                
                my_target[i] = v.cuda()
    
    return my_input, my_target

def map_new_tag(my_target):
    
    a = my_target["target_loss_labels"]
    o = my_target["opinion_loss_labels"]
    s = my_target["polarity_loss_labels"]
    aspect_to_tag_dic = my_target["aspect_to_tag_dic"]
    opinion_to_tag_dic = my_target["opinion_to_tag_dic"]
    sentiment_to_tag_dic = my_target["sentiment_to_tag_dic"]
    tag_to_index = my_target["tag_to_index"] 
    batch_sent_tag_list = []
    batch_a, batch_o, batch_s = list(a), list(o), list(s)
    for a, o, s in zip(batch_a, batch_o, batch_s) :
        each_sent_tag_list = []
        for a_item, o_item, s_item in zip(a, o, s) :
            a_item = int(a_item)
            o_item = int(o_item)
            s_item = int(s_item)
                    

            temp_a_tag = aspect_to_tag_dic[a_item]

            if a_item != 0 and o_item != 0:
                temp_o_tag= "O"
            else:
                temp_o_tag= opinion_to_tag_dic[o_item]
    
            temp_s_tag= sentiment_to_tag_dic[s_item]
                    
            new_tag = temp_a_tag + "_" + temp_o_tag+ "_"+temp_s_tag
            each_sent_tag_list.append(tag_to_index[new_tag])
    
        batch_sent_tag_list.append(each_sent_tag_list)
    
    return torch.tensor(batch_sent_tag_list)

def max_min_normalize(X, range_scope):
    min, max = range_scope
    X_std = (X - torch.min(X, dim=-1)[0].unsqueeze(-1)) / (torch.max(X, dim=-1)[0].unsqueeze(-1) - torch.min(X, dim=-1)[0].unsqueeze(-1))
    X_scaled = X_std * (max - min) + min
    return X*X_scaled

def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                # create_graph=not first_order,
                                create_graph=not not first_order,
                                retain_graph=True,
                                )

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params




