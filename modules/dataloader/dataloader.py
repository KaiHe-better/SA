import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, RandomSampler
from numpy.random import seed
import numpy as np
import random

SEED = 123
seed(SEED)

# from memory_profiler import profile
# @profile(precision=4, stream=open('batch.log','w+')) 

class My_Dataset(object):
    def __init__(self, sent_data, ae_data, oe_data, sc_data, is_testing, max_len, tokenizer):
        self.all_data, self.ae_data, self.oe_data, self.sa_data = sent_data, ae_data, oe_data, sc_data
        self.is_testing=is_testing
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.label_pad_index = 0
        self.ont_hot_map = {0:[1, 0, 0], 1:[0,1,0], 2:[0,0,1]}
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        # index = 37
        raw_tokens = self.all_data[index].strip().split()
        ae_labels = [int(i) for i in self.ae_data[index].strip().split()]
        oe_labels = [int(i) for i in self.oe_data[index].strip().split()]
        sa_labels = [int(i) for i in self.sa_data[index].strip().split()]
        
        split_tokens = []
        split_ae_labels = []
        split_oe_labels = []
        split_sa_labels = []
        for ix, raw_token in enumerate(raw_tokens):
            raw_token = raw_token.lower()
            sub_tokens= self.tokenizer.tokenize(raw_token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if ae_labels[ix]==1 and jx>0:
                    split_ae_labels.append(self.ont_hot_map[2])
                else:
                    split_ae_labels.append(self.ont_hot_map[ae_labels[ix]])

                if oe_labels[ix]==1 and jx>0:
                    split_oe_labels.append(self.ont_hot_map[2])
                else:
                    split_oe_labels.append(self.ont_hot_map[oe_labels[ix]])
                    
                if ix != len(raw_tokens)-1:   
                    split_sa_labels.append(sa_labels[ix])
                else:
                    if len(sub_tokens)>1:
                        if ae_labels[-1] ==0 and jx<len(sub_tokens)-1:
                            split_sa_labels.append(0)
                        else:
                            split_sa_labels.append(sa_labels[ix])
                    else:
                        split_sa_labels.append(sa_labels[ix])

        split_sa_labels_loss = []
        for l in split_sa_labels:
            # 0=background, 1=positive, 2=negative, 3=neutral, 4=conflict.
            if l == 0:
                split_sa_labels_loss.append([1,0,0,0])
            elif l == 1:
                split_sa_labels_loss.append([0,1,0,0])
            elif l == 2:
                split_sa_labels_loss.append([0,0,1,0])
            elif l == 3:
                split_sa_labels_loss.append([0,0,0,1])
            elif l == 4:
                split_sa_labels_loss.append([1,0,0,0])
            else:
                raise ValueError
        
        'Add [CLS] and [SEP] for BERT'
        split_tokens.insert(0, self.tokenizer.bos_token) 
        split_tokens.append(self.tokenizer.eos_token)
        split_ae_labels.insert(0, [1,0,0]) 
        split_oe_labels.insert(0, [1,0,0]) 
        split_sa_labels_loss.insert(0, [1,0,0,0]) 
 
        bert_input_per = self.tokenizer.convert_tokens_to_ids(split_tokens)
        bert_mask_per = [1] * len(bert_input_per)

        pad_label_0 = [[0,0,0]] * (self.max_length - len(split_tokens)+1)
        pad_label_2 = [[0,0,0,0]] * (self.max_length - len(split_tokens)+1)
        pad_label_1 = [0] * (self.max_length - len(split_tokens))
        split_ae_labels = split_ae_labels + pad_label_0
        split_oe_labels = split_oe_labels + pad_label_0
        
        if ae_labels[-1] ==0:
            split_sa_labels_loss[0], split_sa_labels_loss[-1] = split_sa_labels_loss[-1], split_sa_labels_loss[0]
        
        split_sa_labels = split_sa_labels_loss + pad_label_2
        bert_input_per = bert_input_per + pad_label_1
        bert_mask_per = bert_mask_per + pad_label_1
        bert_segment_per = [0] * self.max_length
        
        # return split_ae_labels, split_oe_labels, split_sa_labels, \
        #        sentiment_mask, index,\
        #        bert_input_per, bert_mask_per, bert_segment_per, split_tokens
               
        return index, bert_input_per, bert_mask_per, bert_segment_per, split_tokens, split_ae_labels, split_oe_labels, split_sa_labels

    def collate_fn(self, data):
        aspect_y_raw, opinion_y_raw, sentiment_y_raw, \
        sentiment_mask, index, \
        bert_input, bert_mask, bert_segment, split_tokens = zip(*data)
         
        batch_inputs ={ 'input_ids': torch.LongTensor(bert_input),
                        'attention_mask': torch.LongTensor(bert_mask),
                        'sentiment_mask': torch.LongTensor(sentiment_mask), 
                        'token_type_ids': torch.LongTensor(bert_segment) }
        
        batch_labels ={ 'opinion_loss_labels': torch.LongTensor(opinion_y_raw), 
                       'polarity_loss_labels': torch.LongTensor(sentiment_y_raw), 
                       'target_loss_labels': torch.LongTensor(aspect_y_raw),
                       'index': torch.LongTensor(index),
                       "tokens": split_tokens  }

        return batch_inputs, batch_labels

    def get_tensor_dataset(self):
        index_list, bert_input_per, bert_mask_per, bert_segment_per, ae_labels, oe_labels, sc_labels = [],[],[],[],[],[],[]
        for index in range(len(self.all_data)):
            item = self.__getitem__(index)
            index_list.append(item[0])
            bert_input_per.append(item[1])
            bert_mask_per.append(item[2])
            bert_segment_per.append(item[3])
            ae_labels.append(item[5])
            oe_labels.append(item[6])
            sc_labels.append(item[7])
        
        return TensorDataset(torch.LongTensor(index_list), torch.LongTensor(bert_input_per), torch.LongTensor(bert_mask_per), torch.LongTensor(bert_segment_per), \
                             torch.FloatTensor(ae_labels), torch.FloatTensor(oe_labels), torch.FloatTensor(sc_labels))
                            # torch.FloatTensor(ae_labels), torch.LongTensor(oe_labels), torch.LongTensor(sc_labels))
    
def get_loader(corpus, data_ratio, remain_pesudo_data_ratio, add_other_corpus, batch_size, max_len, tokenizer, num_workers=8):
    train_dev_test_list = ['data/{}/train/'.format(corpus), 'data/{}/dev/'.format(corpus), 'data/{}/test/'.format(corpus)]
    
    data_set_list = []
    for fname in train_dev_test_list:
        sent_data = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
        ae_data = open(fname + r'target.txt', 'r', encoding='utf-8').readlines()
        oe_data = open(fname + r'opinion.txt', 'r', encoding='utf-8').readlines()
        sc_data = open(fname + r'target_polarity.txt', 'r', encoding='utf-8').readlines()
        all_index = list(range(len(sent_data)))
        need_index_list = random.sample(all_index, int(len(sent_data)*data_ratio))
        remain_index_list =  list(set(all_index).difference(set(need_index_list))) 
        if "train" in fname:
            is_testing = False
            need_sent_data = np.array(sent_data)[need_index_list].tolist()
            need_ae_data = np.array(ae_data)[need_index_list].tolist()
            need_oe_data = np.array(oe_data)[need_index_list].tolist()
            need_sc_data = np.array(sc_data)[need_index_list].tolist()
            
            remain_sent_data = np.array(sent_data)[remain_index_list].tolist()
            remain_ae_data = np.array(ae_data)[remain_index_list].tolist()
            remain_oe_data = np.array(oe_data)[remain_index_list].tolist()
            remain_sc_data = np.array(sc_data)[remain_index_list].tolist()
            
            if (remain_pesudo_data_ratio<1):
                all_remain_pesudo_num = int(len(sent_data)* remain_pesudo_data_ratio)
                all_remain_pesudo_index = list(range(len(remain_sent_data)))
                need_remain_index_list = random.sample(all_remain_pesudo_index, all_remain_pesudo_num) 
                remain_sent_data = np.array(remain_sent_data)[need_remain_index_list].tolist()
                remain_ae_data = np.array(remain_ae_data)[need_remain_index_list].tolist()
                remain_oe_data = np.array(remain_oe_data)[need_remain_index_list].tolist()
                remain_sc_data = np.array(remain_sc_data)[need_remain_index_list].tolist()
                
            if add_other_corpus:
                if corpus =="res14":
                    addedd_corpus =  "res15"
                elif corpus =="res15":
                    addedd_corpus =  "res14"
                elif corpus =="lap14":
                    addedd_corpus =  "lap15"
                else:
                    raise Exception()
                pesu_train_dev_test_list = ['data/{}/train/'.format(corpus), 'data/{}/dev/'.format(corpus), 'data/{}/test/'.format(corpus)]
                for fname1 in pesu_train_dev_test_list:
                    remain_sent_data.extend(open(fname1 + r'sentence.txt', 'r', encoding='utf-8').readlines())
                    remain_ae_data.extend(open(fname1 + r'target.txt', 'r', encoding='utf-8').readlines())
                    remain_oe_data.extend(open(fname1 + r'opinion.txt', 'r', encoding='utf-8').readlines())
                    remain_sc_data.extend(open(fname1 + r'target_polarity.txt', 'r', encoding='utf-8').readlines())
        else:       
            is_testing = True
            need_sent_data, need_ae_data, need_oe_data, need_sc_data = sent_data, ae_data, oe_data, sc_data   
        
        gold_data = My_Dataset(need_sent_data, need_ae_data, need_oe_data, need_sc_data, is_testing, max_len, tokenizer)
        gold_dataset = gold_data.get_tensor_dataset()
        data_loader = DataLoader(dataset=gold_dataset,  batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        data_set_list.append(data_loader)
        if "train" in fname:
            train_data_set = gold_dataset
    
    
    if add_other_corpus or len(remain_sent_data)>0:
        pesu_data = My_Dataset(remain_sent_data, remain_ae_data, remain_oe_data, remain_sc_data, False, max_len, tokenizer)
        pesu_dataset = pesu_data.get_tensor_dataset()
        pesu_data_loader = DataLoader(dataset=pesu_dataset,  batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        data_set_list.append(pesu_data_loader)
    else:
        data_set_list.append(None)
        
    return data_set_list, train_data_set




class My_pesuo_Dataset_labled(object):
    def __init__(self, data_list, max_len, tokenizer, now_ae_pesuo_label_list, now_oe_pesuo_label_list, now_sc_pesuo_label_list):
        self.pesuo_sent = data_list
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.label_pad_index = 0

        self.ae_label = now_ae_pesuo_label_list
        self.oe_label = now_oe_pesuo_label_list
        self.sc_label = now_sc_pesuo_label_list

        assert len(self.pesuo_sent) ==len(self.ae_label)

    def __len__(self):
        return len(self.pesuo_sent)
    
    def __getitem__(self, index):
        raw_tokens = self.pesuo_sent[index]
        ae_label = self.ae_label[index]
        oe_label = self.oe_label[index]
        sc_label = self.sc_label[index]

        'Add [CLS] and [SEP] for BERT'
 
        # bert_input_per = self.tokenizer.convert_tokens_to_ids(raw_tokens)
        bert_input_per = raw_tokens.tolist()
        bert_mask_per = [1] * len(raw_tokens)

        pad_label_1 = [0] * (self.max_length - len(bert_input_per))
        pad_label_2 = [[0.0, 0.0, 0.0]] * (self.max_length - len(bert_input_per))
        pad_label_3 = [[0.0, 0.0, 0.0, 0.0]] * (self.max_length - len(bert_input_per))
        bert_input_per = bert_input_per + pad_label_1
        bert_mask_per = bert_mask_per + pad_label_1
        bert_segment_per = [0] * self.max_length
        
        ae_labels = ae_label.tolist()[:len(raw_tokens)] + pad_label_2
        oe_labels = oe_label.tolist()[:len(raw_tokens)] + pad_label_2
        sc_labels = sc_label.tolist()[:len(raw_tokens)] + pad_label_3

        return index, bert_input_per, bert_mask_per, bert_segment_per, raw_tokens, ae_labels, oe_labels, sc_labels

    def collate_fn(self, data):
        index, bert_input, bert_mask, bert_segment, split_tokens, ae_labels, oe_labels, sa_labels = zip(*data)
         
        batch_inputs ={'input_ids': torch.tensor(bert_input),
                        'attention_mask': torch.tensor(bert_mask),
                        'token_type_ids': torch.tensor(bert_segment),
                        }

        batch_labels ={'target_loss_labels': torch.tensor(ae_labels), 
                       'opinion_loss_labels': torch.tensor(oe_labels), 
                       'polarity_loss_labels': torch.tensor(sa_labels),
                       'index': torch.tensor(index),
                       "tokens": split_tokens  
                       }

        return batch_inputs, batch_labels

    def get_tensor_dataset(self):
        index_list, bert_input_per, bert_mask_per, bert_segment_per, ae_labels, oe_labels, sc_labels = [],[],[],[],[],[],[]
        for index in range(len(self.pesuo_sent)):
            item = self.__getitem__(index)
            index_list.append(item[0])
            bert_input_per.append(item[1])
            bert_mask_per.append(item[2])
            bert_segment_per.append(item[3])
            ae_labels.append(item[5])
            oe_labels.append(item[6])
            sc_labels.append(item[7])
        
        return TensorDataset(torch.LongTensor(index_list), torch.LongTensor(bert_input_per), torch.LongTensor(bert_mask_per), torch.LongTensor(bert_segment_per), \
                             torch.FloatTensor(ae_labels), torch.FloatTensor(oe_labels), torch.FloatTensor(sc_labels))
                            #  torch.LongTensor(ae_labels), torch.LongTensor(oe_labels), torch.LongTensor(sc_labels))
        
def get_pesuo_loader_labled(data_list, batch_size, max_length, tokenizer, gold_dataset,
            now_ae_pesuo_label_list, now_oe_pesuo_label_list, now_sc_pesuo_label_list, num_workers=8):

    pesuo_data = My_pesuo_Dataset_labled(data_list, max_length, tokenizer, 
                            now_ae_pesuo_label_list, now_oe_pesuo_label_list, now_sc_pesuo_label_list)
    pesuo_dataset = pesuo_data.get_tensor_dataset()
    concat_dataset = ConcatDataset([pesuo_dataset, gold_dataset])
    pesuo_data_loader = DataLoader(dataset=concat_dataset,  batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return pesuo_data_loader
