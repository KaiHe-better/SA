import os
import sys
import json
from tqdm import tqdm
import shutil
import torch
from .dataloader.dataloader import get_pesuo_loader_labled
from torch import optim
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from collections import deque
from .evaluation import get_metric
import torch.nn.functional as F
from .utils.my_utils import data_to_device, Print_and_write
from apex import amp
from .utils.constrasive import contrastive_loss, R_drop_loss
from .utils.my_utils import gradient_update_parameters

class My_Train_Framework:
    
    def __init__(self, train_ID, train_index, train_data_loader, val_data_loader, test_data_loader, unlabled_data_loader, train_dataset, opt):
        super(My_Train_Framework, self).__init__()
        self.opt = opt
        self.train_index = train_index
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.unlabled_data_loader = unlabled_data_loader
        self.max_len = opt.max_len
        self.train_dataset = train_dataset
        self.if_fp16= opt.if_fp16
        
        if train_ID is None:
            self.res_path ="./results/output/"+str(sys.argv[1:])
        else:
            self.res_path ="./results/output/temp"
            
        if self.train_index==0:
            if os.path.exists(self.res_path):
                shutil.rmtree(self.res_path)
            os.mkdir(self.res_path)
            with open(os.path.join(self.res_path, "config.json"), "w") as f:
                json.dump(vars(opt), f, indent=2)
                
        self.my_print_file = Print_and_write(self.res_path+'/training.txt')
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt, file=self.my_print_file)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def __initialize_stu__(self, stu_model, load_ckpt_path=None, optimizer="adamw", 
                                weight_decay=0, encoder_lr=2e-5,  classifier_lr=1e-4, total_epochs=100):
        print("initialize stu ...")
        self.stu_model = stu_model
        # if load_ckpt_path != "checkpoint/None":
        #     state_dict = self.__load_model__(load_ckpt_path)['state_dict']
        #     own_state = self.stu_model.state_dict()
        #     for name, param in state_dict.items():
        #         if name in own_state.keys():
        #             own_state[name].copy_(param)

        if torch.cuda.is_available():
            self.stu_model.cuda()
        
        if optimizer == 'sgd':
            pytorch_optim = optim.SGD
        elif optimizer == 'adam':
            pytorch_optim = optim.Adam
        elif optimizer == 'adamw':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError

        # pytorch_optim = optim.SGD
        
        encoder_parameters_to_optimize =   list(self.stu_model.sentence_encoder.named_parameters())   
        classifier_parameters_to_optimize = list(self.stu_model.my_classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        parameters_to_optimize = [
            {'params': [p for n, p in encoder_parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, "lr":encoder_lr},
            {'params': [p for n, p in encoder_parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":encoder_lr},
            {'params': [p for n, p in classifier_parameters_to_optimize  if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, "lr":classifier_lr},
            {'params': [p for n, p in classifier_parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":classifier_lr},
            ]
        
        self.stu_optim = pytorch_optim(parameters_to_optimize)
        self.stu_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.stu_optim, num_warmup_steps=self.opt.warmup_step, 
                                                                            num_training_steps=total_epochs*len(self.train_data_loader), num_cycles=3)
        # self.stu_scheduler = ""
        
        if self.if_fp16:
            self.stu_model, self.stu_optim = amp.initialize(self.stu_model, self.stu_optim, opt_level='O1')
            
    def __initialize_tea__(self, tea_model, load_ckpt_path=None, optimizer="adamw", 
                                weight_decay=0, encoder_lr=2e-5, classifier_lr=1e-4, total_epochs=100):
        print("initialize tea ...")
        self.tea_model = tea_model
        # if load_ckpt_path != "checkpoint/None":
        #     state_dict = self.__load_model__(load_ckpt_path)['state_dict']
        #     own_state = self.tea_model.state_dict()
        #     for name, param in state_dict.items():
        #         if name in own_state.keys():
        #             own_state[name].copy_(param)

        if torch.cuda.is_available():
            self.tea_model.cuda()
        
        if optimizer == 'sgd':
            pytorch_optim = optim.SGD
        elif optimizer == 'adam':
            pytorch_optim = optim.Adam
        elif optimizer == 'adamw':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError

        encoder_parameters_to_optimize =   list(self.tea_model.sentence_encoder.named_parameters())   
        classifier_parameters_to_optimize = list(self.tea_model.my_classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        parameters_to_optimize = [
            {'params': [p for n, p in encoder_parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, "lr":encoder_lr},
            {'params': [p for n, p in encoder_parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":encoder_lr},
            {'params': [p for n, p in classifier_parameters_to_optimize  if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, "lr":classifier_lr},
            {'params': [p for n, p in classifier_parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":classifier_lr},
            ]
        
        self.tea_optim = pytorch_optim(parameters_to_optimize)
        self.tea_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.tea_optim, num_warmup_steps=self.opt.warmup_step, 
                                                                            num_training_steps=total_epochs*len(self.train_data_loader), num_cycles=3)
        # self.tea_scheduler = ""
        
        if self.if_fp16:
            self.tea_model, self.tea_optim = amp.initialize(self.tea_model, self.tea_optim, opt_level='O1')
            
    def __initialize_weighter__(self, my_weighter, load_ckpt_path=None, optimizer="adamw", 
                                weight_decay=0, weight_lr=1e-4, total_epochs=100):
        print("initialize weighter ...")
        self.my_weighter = my_weighter
        # if load_ckpt_path != "checkpoint/None":
        #     state_dict = self.__load_model__(load_ckpt_path)['state_dict']
        #     own_state = self.tea_model.state_dict()
        #     for name, param in state_dict.items():
        #         if name in own_state.keys():
        #             own_state[name].copy_(param)

        if torch.cuda.is_available():
            self.my_weighter.cuda()
        
        if optimizer == 'sgd':
            pytorch_optim = optim.SGD
        elif optimizer == 'adam':
            pytorch_optim = optim.Adam
        elif optimizer == 'adamw':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError
        
        my_weighter_parameters_to_optimize =   list(self.my_weighter.named_parameters())   
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        parameters_to_optimize = [
            {'params': [p for n, p in my_weighter_parameters_to_optimize  if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, "lr":weight_lr},
            {'params': [p for n, p in my_weighter_parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":weight_lr},
            ]
        
        self.weighter_optim = pytorch_optim(parameters_to_optimize)
        self.weighter_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.weighter_optim, num_warmup_steps=self.opt.warmup_step, 
                                                                            num_training_steps=total_epochs*len(self.train_data_loader), num_cycles=3)
        # self.weighter_scheduler = ""
        
        if self.if_fp16:
            self.my_weighter, self.weighter_optim = amp.initialize(self.my_weighter, self.weighter_optim, opt_level='O1')
            
    def get_measure(self, train_flag, epoch_input_list, epoch_target_list, epoch_pred_list, epoch=0, record_flag=False, res_list=None):
        index_list, token_list = [], []
        a_labels, a_preds, o_labels, o_preds, s_labels, s_preds, final_mask =  [], [], [], [], [], [], []
        for batch_input_dic, batch_target_dic, batch_pred_dic in zip(epoch_input_list, epoch_target_list, epoch_pred_list):
            for index,  attention_mask, target_lables, opinion_lables, polarity_lables, logits_ae, logits_sc, logits_oe in \
                zip(batch_target_dic["index"], 
                    batch_input_dic["attention_mask"], 
                    batch_target_dic["target_loss_labels"], 
                    batch_target_dic["opinion_loss_labels"], 
                    batch_target_dic["polarity_loss_labels"], 
                    batch_pred_dic["logits_ae"], 
                    batch_pred_dic["logits_sc"], 
                    batch_pred_dic["logits_oe"],
                    ):
                
                index_list.append(index)
                token_list.append("-")
                
                a_labels.append(target_lables)
                a_preds.append(torch.argmax(logits_ae, dim=-1))
                
                o_labels.append(opinion_lables)
                o_preds.append(torch.argmax(logits_oe, dim=-1))
                
                s_labels.append(polarity_lables)
                s_preds.append(torch.argmax(logits_sc, dim=-1))
                
                final_mask.append(attention_mask)
        
        aspect_PRF, opinion_PRF, sentiment_PRF, ABSA_PRF \
                    = get_metric(train_flag, self.res_path, index_list, token_list, 
                                 a_labels,  o_labels, s_labels,
                                 a_preds, o_preds, s_preds, final_mask, epoch, record_flag, res_list)
        
        dic_F = {}
        dic_F["AE_P"] = aspect_PRF[0]*100
        dic_F["AE_R"] = aspect_PRF[1]*100
        dic_F["AE_F"] = aspect_PRF[2]*100
        dic_F["OE_P"] = opinion_PRF[0]*100
        dic_F["OE_R"] = opinion_PRF[1]*100
        dic_F["OE_F"] = opinion_PRF[2]*100
        dic_F["SC_P"] = sentiment_PRF[0]*100
        dic_F["SC_R"] = sentiment_PRF[1]*100
        dic_F["SC_F"] = sentiment_PRF[2]*100
        dic_F["ABSA_P"] = ABSA_PRF[0]*100
        dic_F["ABSA_R"] = ABSA_PRF[1]*100
        dic_F["ABSA_F"] = ABSA_PRF[2]*100
        return dic_F
    
    def chosed_pesudo_label(self, total_ae, total_oe, total_sc, gold_res_list):
        chosed_data_index, total_ae, total_oe, total_sc  = self.weight_pesuo(total_ae, total_oe, total_sc, 
                                                  chosed_th=self.opt.chosed_th, uncertain_th_S=self.opt.uncertain_th_S, uncertain_th_E=self.opt.uncertain_th_E)
        
        # ae_label_weights, total_ae_pesuo_label = torch.max(total_ae, dim=2)  
        # oe_label_weights, total_oe_pesuo_label = torch.max(total_oe, dim=2)
        # sc_label_weights, total_sc_pesuo_label = torch.max(total_sc, dim=2)
        
        total_ae_pesuo_label = total_ae
        total_oe_pesuo_label = total_oe
        total_sc_pesuo_label = total_sc
        
        total_gold_sent, total_ae_gold_label, total_oe_gold_label, total_sc_gold_label = [],[],[],[]
        for input_item, target_item in gold_res_list:
            total_gold_sent.extend(input_item["input_ids"])
            total_ae_gold_label.append(target_item["target_loss_labels"])
            total_oe_gold_label.append(target_item["opinion_loss_labels"])
            total_sc_gold_label.append(target_item["polarity_loss_labels"])

        total_ae_gold_label = torch.cat(total_ae_gold_label, dim=0)
        total_oe_gold_label = torch.cat(total_oe_gold_label, dim=0)
        total_sc_gold_label = torch.cat(total_sc_gold_label, dim=0)
        
        chosed_data_sent, chosed_ae_label, chosed_oe_label, chosed_sc_label = [], [], [] , []
        gold_chosed_ae_label, gold_chosed_oe_label, gold_chosed_sc_label = [], [] , []
        for index in chosed_data_index:
            chosed_data_sent.append(total_gold_sent[index])
            chosed_ae_label.append(total_ae_pesuo_label[index])
            chosed_oe_label.append(total_oe_pesuo_label[index])
            chosed_sc_label.append(total_sc_pesuo_label[index])

            gold_chosed_ae_label.append(total_ae_gold_label[index])
            gold_chosed_oe_label.append(total_oe_gold_label[index])
            gold_chosed_sc_label.append(total_sc_gold_label[index])

        mix_data_loader = get_pesuo_loader_labled(chosed_data_sent, self.opt.batch_size, self.max_len, self.tokenizer, self.train_dataset,
                                        chosed_ae_label, chosed_oe_label, chosed_sc_label)

        print("\n mix_data_loader: {} / gold_data_loader: {} / unlabled_data_loader: {} \n".format(len(mix_data_loader), len(self.train_data_loader), \
                                                     len(self.unlabled_data_loader)), file=self.my_print_file)

        return mix_data_loader, (gold_chosed_ae_label, gold_chosed_oe_label, gold_chosed_sc_label)

    def weight_pesuo(self, total_ae, total_oe, total_sc, chosed_th=0.5, uncertain_th_S=0.2, uncertain_th_E=0.8):
        
        ae = F.softmax(total_ae, dim=-1)
        oe = F.softmax(total_oe, dim=-1)
        sc = F.softmax(total_sc, dim=-1)
        
        ae_chose =  torch.nonzero(ae[:, :, 1:]>chosed_th)[:, 0].tolist()
        oe_chose =  torch.nonzero(oe[:, :, 1:]>chosed_th)[:, 0].tolist()
        sc_chose =  torch.nonzero(sc[:, :, 1:]>chosed_th)[:, 0].tolist()
        chosed_data_index = list(set(ae_chose).union(set(sc_chose)).union(set(oe_chose)))
        
        add_data_len_S = int(total_ae.size()[0] *uncertain_th_S)
        add_data_len_E = int(total_ae.size()[0] *uncertain_th_E)
        ae_uncertain = torch.sum(torch.mul(-ae, torch.log(ae)), dim=-1)
        oe_uncertain = torch.sum(torch.mul(-oe, torch.log(oe)), dim=-1)
        sc_uncertain = torch.sum(torch.mul(-sc, torch.log(sc)), dim=-1)
        uncertain = torch.mean(ae_uncertain + oe_uncertain + sc_uncertain, dim=1)
        uncertain_chosed_index = torch.sort(uncertain)[1][add_data_len_S:add_data_len_E].tolist()
        
        total_chosed_data_index = list(set(chosed_data_index).intersection(set(uncertain_chosed_index)))
        # total_chosed_data_index = chosed_data_index
        # total_chosed_data_index = uncertain_chosed_index
        
        return total_chosed_data_index, ae, oe, sc
        
    def train_stu_gold_data(self):
        self.stu_model.train()
        with tqdm(total=len(self.train_data_loader), desc="train_stu_gold_data", ncols=170) as pbar:  
            for tensor_data_list in self.train_data_loader:
                my_input, my_target = data_to_device(tensor_data_list)
                self.stu_optim.zero_grad()
                
                final_weight_dic_res = self.stu_model(my_input, my_target)
                stu_gold_loss= final_weight_dic_res["total_loss"]
                if self.if_fp16:
                    with amp.scale_loss(stu_gold_loss, self.stu_optim) as stu_gold_loss:
                        stu_gold_loss.backward()
                else:
                    stu_gold_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), 10.0)
                self.stu_optim.step()
                
                
                if type(self.stu_scheduler) != str:
                    self.stu_scheduler.step()
            
                postfix= {}
                postfix['stu_gold_loss']= float(stu_gold_loss)
                pbar.set_postfix(postfix)
                pbar.update(1)
                
                if self.opt.break_flag:
                    break

        return stu_gold_loss
            
    def train_tea_gold_data(self):
        self.tea_model.train()
        with tqdm(total=len(self.train_data_loader), desc="train_tea_gold_data", ncols=170) as pbar:  
            for tensor_data_list in self.train_data_loader:
                my_input, my_target = data_to_device(tensor_data_list)
                self.tea_optim.zero_grad()
                    
                tea_dic = self.tea_model(my_input, my_target)
                tea_gold_loss= tea_dic["total_loss"]
                if self.if_fp16:
                    with amp.scale_loss(tea_gold_loss, self.tea_optim) as tea_gold_loss:
                        tea_gold_loss.backward()
                else:
                    tea_gold_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.tea_model.parameters(), 10.0)
                self.tea_optim.step()
                
                if type(self.tea_scheduler) != str:
                    self.tea_scheduler.step()
            
                postfix= {}
                postfix['tea_gold_loss']= '{0:.4f}'.format(tea_gold_loss)
                pbar.set_postfix(postfix)
                pbar.update(1)
            
                if self.opt.break_flag:
                    break

        return tea_gold_loss

    def generate_mix_data_loader(self):
        self.tea_model.eval()
        total_logits_ae_list, total_logits_oe_list, total_logits_sc_list, total_gold_res_list = [], [], [], []
        with torch.no_grad():
            with tqdm(total=len(self.unlabled_data_loader), desc="predict_pesuo", ncols=170) as pbar:  
                for tensor_data_list in self.unlabled_data_loader:
                    logits_ae_list, logits_oe_list, logits_sc_list = [],[],[]
                    my_input, my_target = data_to_device(tensor_data_list)
                    total_gold_res_list.append((my_input, my_target))
                    logits_ae, logits_oe, logits_sc = self.tea_model(my_input, if_get_loss=False)
                        
                    total_logits_ae_list.append(logits_ae)
                    total_logits_oe_list.append(logits_oe)
                    total_logits_sc_list.append(logits_sc)
                    
                    postfix= {}
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    
                    if self.opt.break_flag:
                        break

        logits_ae_list = torch.cat(total_logits_ae_list, dim=0)
        logits_oe_list = torch.cat(total_logits_oe_list, dim=0)
        logits_sc_list = torch.cat(total_logits_sc_list, dim=0)

        mix_data_loader, gold_res = self.chosed_pesudo_label(logits_ae_list, logits_oe_list, logits_sc_list, total_gold_res_list)
      
        return mix_data_loader, gold_res
    
    def weig_train_stu_all_data(self, mix_data_loader, gold_res, current_epoch):
        self.stu_model.train()
        meta_loader = iter(self.train_data_loader)
        mix_epoch_input_list, mix_epoch_target_list, mix_epoch_pred_list, weig_list = [],[],[],[]
        with tqdm(total=len(mix_data_loader), desc="train_stu_all_data", ncols=170) as pbar:  
            for tensor_data_list in mix_data_loader:
                
                # step 1 psuedo train stu network with weighted loss
                mix_my_input, mix_my_target = data_to_device(tensor_data_list)
                self.stu_model.zero_grad()
                self.my_weighter.zero_grad()   
                copy_stu_mix_dic = self.stu_model(mix_my_input, mix_my_target, pesuo_flag=True)
                weighted_stu_psuedo_loss, _ = self.my_weighter(copy_stu_mix_dic, mix_my_target, current_epoch)
                meta_stu_params = gradient_update_parameters(self.stu_model, weighted_stu_psuedo_loss, step_size=self.stu_optim.param_groups[0]["lr"])
                weighted_stu_psuedo_loss.backward()
                
                
                # step 2 train my_weighter with gold data
                try:
                    gold_input, gold_target = data_to_device(next(meta_loader))
                    self.stu_model.zero_grad()
                    self.weighter_optim.zero_grad()   
                    copy_stu_gold_dic = self.stu_model(gold_input, gold_target, params=meta_stu_params)
                    weighted_wig_psuedo_loss, _ = self.my_weighter(copy_stu_gold_dic, gold_target, current_epoch )
                    
                    if self.if_fp16:
                        with amp.scale_loss(weighted_wig_psuedo_loss, self.stu_optim) as weighted_wig_psuedo_loss:
                            weighted_wig_psuedo_loss.backward()
                    else:
                        weighted_wig_psuedo_loss.backward()
                        
                    self.weighter_optim.step()
                    if type(self.weighter_scheduler) != str:
                        self.weighter_scheduler.step()
                except StopIteration:
                    pass
                
                # step 3  formally train stu with weighted loss
                self.stu_optim.zero_grad()
                self.weighter_optim.zero_grad()
                stu_mix_dic = self.stu_model(mix_my_input, mix_my_target, pesuo_flag=True)
                weighted_total_loss, total_loss_weight = self.my_weighter(stu_mix_dic, mix_my_target, current_epoch)
                weig_list.append(total_loss_weight)
                
                if self.if_fp16:
                    with amp.scale_loss(weighted_total_loss, self.stu_optim) as weighted_total_loss:
                        weighted_total_loss.backward()
                else:
                    weighted_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), 10)
                self.stu_optim.step()
             
                mix_epoch_input_list.append(mix_my_input)
                mix_epoch_target_list.append(mix_my_target) 
                mix_epoch_pred_list.append(stu_mix_dic)
                
                postfix= {}
                postfix['total_loss']= '{0:.4f}'.format(weighted_total_loss) 
                pbar.set_postfix(postfix)
                pbar.update(1)

                if self.opt.break_flag:
                    break

        mix_train_dic_F = self.get_measure("Train", mix_epoch_input_list, mix_epoch_target_list, mix_epoch_pred_list, record_flag="Pesuo", res_list=gold_res)  
        print("\n total_loss:{:.4f}".format(weighted_total_loss), file=open(self.res_path+'/training.txt', "a", encoding='utf-8')) 
        torch.cuda.empty_cache()    
        return mix_train_dic_F, mix_epoch_pred_list, weig_list
        
    def no_weig_train_stu_all_data(self, mix_data_loader, gold_res):
        self.stu_model.train()
        mix_epoch_input_list, mix_epoch_target_list, mix_epoch_pred_list = [],[],[]
        with tqdm(total=len(mix_data_loader), desc="train_stu_all_data", ncols=170) as pbar:  
            for tensor_data_list in mix_data_loader:
                mix_my_input, mix_my_target = data_to_device(tensor_data_list)
                
                stu_mix_dic = self.stu_model(mix_my_input, mix_my_target, pesuo_flag=True)
                stu_mix_loss= stu_mix_dic["total_loss"] * self.opt.mix_loss_weight 
                
                if self.opt.if_constra:
                    tea_final_weight_dic_res = self.stu_model(mix_my_input, mix_my_target)
                    tea_mix_loss= tea_final_weight_dic_res["total_loss"] * self.opt.mix_loss_weight 

                    ae_contra_loss = R_drop_loss(stu_mix_dic["logits_ae"], tea_final_weight_dic_res["logits_ae"])
                    oe_contra_loss = R_drop_loss(stu_mix_dic["logits_oe"], tea_final_weight_dic_res["logits_oe"])
                    sc_contra_loss = R_drop_loss(stu_mix_dic["logits_sc"], tea_final_weight_dic_res["logits_sc"])
                    mix_contra_loss = torch.sum(ae_contra_loss+oe_contra_loss+sc_contra_loss)/3*self.opt.contra_loss_weight
                    total_loss = (stu_mix_loss + tea_mix_loss) /2 + mix_contra_loss
                else:
                    mix_contra_loss = -1
                    total_loss = stu_mix_loss
                    
                if self.if_fp16:
                    with amp.scale_loss(total_loss, self.stu_optim) as total_loss:
                        total_loss.backward()
                else:
                    total_loss.backward()
                    
                torch.nn.utils.clip_grad_norm_(self.stu_model.parameters(), 10)
                self.stu_optim.step()
                self.stu_optim.zero_grad()
                
                mix_epoch_input_list.append(mix_my_input)
                mix_epoch_target_list.append(mix_my_target) 
                mix_epoch_pred_list.append(stu_mix_dic)
                
                postfix= {}
                postfix['total_loss']= '{0:.4f}'.format(total_loss) 
                postfix['stu_mix_loss']= '{0:.4f}'.format(stu_mix_loss) 
                if self.opt.if_weight_loss:
                    postfix['weight']= '{0:.2f}/{0:.2f}/{0:.2f}'.format(stu_mix_dic["ae_loss_weight"], stu_mix_dic["oe_loss_weight"], stu_mix_dic["sc_loss_weight"])
                pbar.set_postfix(postfix)
                pbar.update(1)

                if self.opt.break_flag:
                    break
                            
        mix_train_dic_F = self.get_measure("Train", mix_epoch_input_list, mix_epoch_target_list, mix_epoch_pred_list, record_flag="Pesuo", res_list=gold_res)  
        print("\n total_loss:{:.4f}".format(total_loss), file=open(self.res_path+'/training.txt', "a", encoding='utf-8'))    
        return mix_train_dic_F, mix_epoch_pred_list
    
    def train_no_self_training(self, tea_model, stu_model, my_weighter, optim_name, save_ckpt_path, load_ckpt_path, corpus=None,
              if_weight_loss=False, total_epochs=1000,  weight_decay=0, 
              encoder_lr=2e-5, weight_lr=2e-5, classifier_lr=2e-5):
        
        self.tokenizer = tea_model.sentence_encoder.tokenizer
        self.__initialize_stu__(stu_model, load_ckpt_path, optim_name, weight_decay, encoder_lr, classifier_lr, total_epochs)
        print("\n", file=self.my_print_file)
        print("Start training...", file=self.my_print_file)
        print("\n", file=self.my_print_file)
        
        best_dic_F = {}
        best_f_absa = -1 
        best_epoch = 0 
        for Total in range(total_epochs):
            print("Total epoch : {}".format(Total), file=self.my_print_file)
            stu_gold_loss = self.train_stu_gold_data()
            val_dic_F = self.eval("Test", self.stu_model, 0, False)
            print("stu_gold_loss:{0:.4f}, val_ABSA_F:{1:.4f}".format(stu_gold_loss, val_dic_F["ABSA_F"]), file=open(self.res_path+'/training.txt', "a", encoding='utf-8'), end="")
            
            if val_dic_F["ABSA_F"] > best_f_absa:
                print("save ckpt !", file=self.my_print_file)
                if if_weight_loss:
                    torch.save({'state_dict': self.my_weighter.state_dict()}, save_ckpt_path)
                best_f_absa = val_dic_F["ABSA_F"]
                best_epoch = Total
                best_dic_F = val_dic_F
                
            with open(self.res_path+"/"+str(self.train_index)+"_performanc.txt", "a") as f:
                f.write("valid_AE_P:   {0:.2f},  AE_R: {1:.2f},   AE_F:   {2:.2f}".format(float(val_dic_F["AE_P"]), float(val_dic_F["AE_R"]), float(val_dic_F["AE_F"]))+"\r")
                f.write("valid_OE_P:   {0:.2f},  OE_R: {1:.2f},   OE_F:   {2:.2f}".format(float(val_dic_F["OE_P"]), float(val_dic_F["OE_R"]), float(val_dic_F["OE_F"]))+"\r")
                f.write("valid_SC_P:   {0:.2f},  SC_R: {1:.2f},   SC_F:   {2:.2f}".format(float(val_dic_F["SC_P"]), float(val_dic_F["SC_R"]), float(val_dic_F["SC_F"]))+"\r")
                f.write("valid_ABSA_P: {0:.2f},  ABSA_R: {1:.2f}, ABSA_F: {2:.2f}".format(float(val_dic_F["ABSA_P"]), float(val_dic_F["ABSA_R"]), float(val_dic_F["ABSA_F"]))+"\r")
                f.write("\n\r")
                f.write("best epoch {0}, ABSA_F {1:.2f} ".format(best_epoch, best_f_absa))
                f.write("\r===========================================================")
                f.write("\n")
                f.write("\n")
            print("best epoch {0}, best_f_absa {1:.2f} \n ".format(best_epoch, best_f_absa), file=self.my_print_file)
            print("\n\n\r", file=self.my_print_file)
            
        print("Finish !", file=self.my_print_file)
        return best_epoch, best_dic_F           
            
    def train(self, tea_model, stu_model, my_weighter, optim_name, save_ckpt_path, load_ckpt_path, corpus=None,
              if_weight_loss=False, total_epochs=1000,  weight_decay=0, 
              encoder_lr=2e-5, weight_lr=2e-5, classifier_lr=2e-5):
        
        self.tokenizer = tea_model.sentence_encoder.tokenizer
        self.__initialize_tea__(tea_model, load_ckpt_path, optim_name, weight_decay, encoder_lr, classifier_lr, total_epochs)
        self.__initialize_stu__(stu_model, load_ckpt_path, optim_name, weight_decay, encoder_lr, classifier_lr, total_epochs)
        if if_weight_loss:
            self.__initialize_weighter__(my_weighter, load_ckpt_path, optim_name, weight_decay, weight_lr, total_epochs)

        print("\n", file=self.my_print_file)
        print("Start training...", file=self.my_print_file)
        print("\n", file=self.my_print_file)
        
        ae_loss_list, oe_loss_list, sc_loss_list, total_weig_list = [],[],[],[]
        best_dic_F = {}
        best_f_absa = -1 
        best_epoch = 0 
        for Total in range(total_epochs):
            print("Total epoch : {}".format(Total), file=self.my_print_file)
            
            for _ in range(self.opt.inner_epoch):
                tea_gold_loss = self.train_tea_gold_data()
                if self.opt.break_flag:
                        break
            val_dic_F = self.eval("Test", self.tea_model, 0, False)
            print("tea_gold_loss:{:.4f}, val_ABSA_F:{:.4f}".format(tea_gold_loss, val_dic_F["ABSA_F"]), file=open(self.res_path+'/training.txt', "a", encoding='utf-8'))


            for _ in range(self.opt.inner_epoch):
                stu_gold_loss = self.train_stu_gold_data()
                if self.opt.break_flag:
                        break
            val_dic_F= self.eval("Test", self.stu_model, 0, False) 
            print("stu_gold_loss:{0:.4f}, val_ABSA_F:{1:.4f}".format(stu_gold_loss, val_dic_F["ABSA_F"]), file=open(self.res_path+'/training.txt', "a", encoding='utf-8'), end="")

            pesuo_data_loader_with_label, gold_res = self.generate_mix_data_loader()
            if if_weight_loss:
                mix_train_dic_F, mix_epoch_pred_list, weig_list = self.weig_train_stu_all_data(pesuo_data_loader_with_label, gold_res, Total) 
            else:
                weig_list = []
                mix_train_dic_F, mix_epoch_pred_list = self.no_weig_train_stu_all_data(pesuo_data_loader_with_label, gold_res) 
            stu_val_dic_F = self.eval("Test", self.stu_model, 0, False) 
            total_weig_list.append(weig_list)
            
            print("train AE: {:.2f}".format(mix_train_dic_F["AE_F"]), file=self.my_print_file)
            print("train OE: {:.2f}".format(mix_train_dic_F["OE_F"]), file=self.my_print_file) 
            print("train SC: {:.2f}".format(mix_train_dic_F["SC_F"]), file=self.my_print_file) 
            print("train ABSA: {:.2f}".format(mix_train_dic_F["ABSA_F"]), file=self.my_print_file)
            
            
            for i in mix_epoch_pred_list:
                ae_loss_list.append(torch.mean(i["loss_ae"], dim=-1).tolist())
                oe_loss_list.append(torch.mean(i["loss_oe"], dim=-1).tolist())
                sc_loss_list.append(torch.mean(i["loss_sc"], dim=-1).tolist())
            
            
            if stu_val_dic_F["ABSA_F"] > best_f_absa:
                print("save ckpt !", file=self.my_print_file)
                if if_weight_loss:
                    torch.save({'state_dict': self.my_weighter.state_dict()}, save_ckpt_path+"_"+str(Total))
                best_f_absa = stu_val_dic_F["ABSA_F"]
                best_epoch = Total
                best_dic_F = stu_val_dic_F
                
                
            with open(self.res_path+"/"+str(self.train_index)+"_performanc.txt", "a") as f:
                    f.write("train_epoch:  "+str(Total+1)+"\r")
                    f.write("train_AE_P:   {0:.2f},  AE_R: {1:.2f},   AE_F:   {2:.2f} ".format(float(mix_train_dic_F["AE_P"]), float(mix_train_dic_F["AE_R"]), float(mix_train_dic_F["AE_F"]))+"\r")
                    f.write("train_OE_P:   {0:.2f},  OE_R: {1:.2f},   OE_F:   {2:.2f} ".format(float(mix_train_dic_F["OE_P"]), float(mix_train_dic_F["OE_R"]), float(mix_train_dic_F["OE_F"]))+"\r")
                    f.write("train_SC_P:   {0:.2f},  SC_R: {1:.2f},   SC_F:   {2:.2f} ".format(float(mix_train_dic_F["SC_P"]), float(mix_train_dic_F["SC_R"]), float(mix_train_dic_F["SC_F"]))+"\r")
                    f.write("train_ABSA_P: {0:.2f},  ABSA_R: {1:.2f}, ABSA_F: {2:.2f} ".format(float(mix_train_dic_F["ABSA_P"]), float(mix_train_dic_F["ABSA_R"]), float(mix_train_dic_F["ABSA_F"]))+"\r")
                    f.write("\n\r")
                    # if if_weight_loss:
                    #     f.write("weight_AE: {0:.2f},  weight_OE: {1:.2f}, weight_sc: {2:.2f}".format(stu_mix_dic["ae_loss_weight"], stu_mix_dic["oe_loss_weight"], stu_mix_dic["sc_loss_weight"])+"\r")
                    #     f.write("\n\r")
                    f.write("valid_AE_P:   {0:.2f},  AE_R: {1:.2f},   AE_F:   {2:.2f}".format(float(stu_val_dic_F["AE_P"]), float(stu_val_dic_F["AE_R"]), float(stu_val_dic_F["AE_F"]))+"\r")
                    f.write("valid_OE_P:   {0:.2f},  OE_R: {1:.2f},   OE_F:   {2:.2f}".format(float(stu_val_dic_F["OE_P"]), float(stu_val_dic_F["OE_R"]), float(stu_val_dic_F["OE_F"]))+"\r")
                    f.write("valid_SC_P:   {0:.2f},  SC_R: {1:.2f},   SC_F:   {2:.2f}".format(float(stu_val_dic_F["SC_P"]), float(stu_val_dic_F["SC_R"]), float(stu_val_dic_F["SC_F"]))+"\r")
                    f.write("valid_ABSA_P: {0:.2f},  ABSA_R: {1:.2f}, ABSA_F: {2:.2f}".format(float(stu_val_dic_F["ABSA_P"]), float(stu_val_dic_F["ABSA_R"]), float(stu_val_dic_F["ABSA_F"]))+"\r")
                    f.write("\n\r")
                    f.write("best epoch {0}, ABSA_F {1:.2f} ".format(best_epoch, best_f_absa))
                    f.write("\r===========================================================")
                    f.write("\n")
                    f.write("\n")
            print("best epoch {0}, best_f_absa {1:.2f} \n ".format(best_epoch, best_f_absa), file=self.my_print_file)
            print("\n\n\r", file=self.my_print_file)
        
        with open(self.res_path+"/"+str(self.train_index)+"_loss.txt", "w") as f:
             f.write(str(ae_loss_list))
             f.write("\n")
             f.write(str(oe_loss_list))
             f.write("\n")
             f.write(str(sc_loss_list))
             
        with open(self.res_path+"/"+str(self.train_index)+"_weights.txt", "w") as f:    
            temp_ae,temp_oe,temp_sc = [],[],[]
            for iii in total_weig_list:
                for i in iii:
                    temp_ae.append(i[0].tolist())
                    temp_oe.append(i[1].tolist())
                    temp_sc.append(i[2].tolist())
            f.write(str(temp_ae))
            f.write("\n")
            f.write(str(temp_oe))
            f.write("\n")
            f.write(str(temp_sc))
            
        print("Finish !", file=self.my_print_file)
        return best_epoch, best_dic_F

    def eval(self, train_flag, model, epoch=0, if_fp16=False):
        model.eval()
        if train_flag == "Valid":
            used_data_loader = self.val_data_loader
        else:
            used_data_loader = self.test_data_loader
        
        with torch.no_grad():
            with tqdm(total=len(used_data_loader), desc=train_flag, ncols=170) as pbar: 
                epoch_input_list, epoch_target_list,epoch_pred_list = [], [], []
                for data in used_data_loader:
                    my_input, my_target = data_to_device(data)
                    dic_res = model(my_input, my_target)
                    epoch_input_list.append(my_input)
                    epoch_target_list.append(my_target)
                    epoch_pred_list.append(dic_res)
                    
                    postfix= {}
                    postfix['total_loss']= '{0:.4f}'.format(dic_res["total_loss"])
                    pbar.set_postfix(postfix) 
                    pbar.update(1)
                
                    if self.opt.break_flag:
                        break
                    
                dic_F = self.get_measure(train_flag, epoch_input_list, epoch_target_list, epoch_pred_list, epoch=epoch)
                postfix['total_loss']= '{0:.2f}'.format(dic_res["total_loss"])
                postfix['AE']= '{0:.2f}'.format(dic_F["AE_F"])
                postfix['OE']= '{0:.2f}'.format(dic_F["OE_F"])
                postfix['SC']= '{0:.2f}'.format(dic_F["SC_F"])
                postfix['ABSA']= '{0:.2f}'.format(dic_F["ABSA_F"])
                pbar.set_postfix(postfix) 
                pbar.update()
                
        print("", file=self.my_print_file)        
        model.train()
        torch.cuda.empty_cache()        
        return dic_F