import torch
import torch.nn as nn
from utils.my_utils import max_min_normalize
# from modules.models.feature_weight_module import Feature_weight_module
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaLinear
                           
class My_weighter(MetaModule):
    def __init__(self, max_len, my_min, my_max):
        super(My_weighter, self).__init__()
        self.my_min, self.my_max, self.max_len = my_min, my_max, max_len
        # self.fc1 = MetaLinear(max_len*3, max_len)
        # self.fc2 = MetaLinear(max_len, 3)
        self.fc1 = MetaLinear(max_len*3, max_len*3)
        self.fc2 = MetaLinear(max_len*3, max_len*3)
        
        
    def forward(self, res_dic, my_target, params=None):
        loss_target, loss_opinion, loss_polarity = res_dic["loss_ae"], res_dic["loss_oe"], res_dic["loss_sc"]
        total_loss = torch.cat((loss_target, loss_opinion, loss_polarity), dim=-1)
        total_loss_weight = self.weight_loss_layer(total_loss, params=self.get_subdict(params, 'weight_loss_layer'))
        normalized_total_loss_weight = max_min_normalize(total_loss_weight, (self.my_min, self.my_max))
        weight_loss = self.get_weighted_totoal_loss(normalized_total_loss_weight, res_dic)
        
        return weight_loss, normalized_total_loss_weight
    
    def weight_loss_layer(self, input_embeds, params=None):
        losits = self.fc1(input_embeds, params=self.get_subdict(params, 'fc1'))
        losits = F.relu(losits)
        losits = self.fc2(losits, params=self.get_subdict(params, 'fc2'))
        losits = F.sigmoid(losits).view(-1, 3*self.max_len)
        return losits
    
    def get_weighted_totoal_loss(self, weights, res_dic):
        loss_ae = torch.mean(weights[:, 0:100] * res_dic["loss_ae"]) # weights = (batch_size, 3) res_dic["loss_ae"] = (batch_size, sent_len)
        loss_oe = torch.mean(weights[:, 100:200] * res_dic["loss_oe"])
        loss_sc = torch.mean(weights[:, 200:300] * res_dic["loss_sc"])
        copy_loss = loss_ae+loss_oe+loss_sc
        
       
        return copy_loss

class My_classifier_model(MetaModule):
    def __init__(self, encoder_hidden_size):
        super(My_classifier_model, self).__init__()

        self.linear_ae = MetaLinear(encoder_hidden_size, 3)
        self.linear_oe = MetaLinear(encoder_hidden_size, 3)
        self.linear_sc = MetaLinear(encoder_hidden_size, 4)
        
    
    def forward(self, logits_ae, logits_oe, logits_sc, params=None):
        logits_ae = self.linear_ae(nn.Dropout(p=0.1)(logits_ae), params=self.get_subdict(params, 'linear_ae'))
        logits_oe = self.linear_oe(nn.Dropout(p=0.1)(logits_oe), params=self.get_subdict(params, 'linear_oe'))
        logits_sc = self.linear_sc(nn.Dropout(p=0.1)(logits_sc), params=self.get_subdict(params, 'linear_sc'))
        
        return logits_ae, logits_oe, logits_sc

    
class My_model(MetaModule):
    def __init__(self, sentence_encoder, if_weight_loss, max_length, my_min=0.9, my_max=3):
        super(My_model, self).__init__()
        self.loss_fn_ae = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,10,10]).float())
        self.loss_fn_oe = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,10,10]).float())
        self.loss_fn_sc = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,10,10,10]).float())
        
        self.pesuo_loss_fn_ae = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([1,10,10]).float())
        self.pesuo_loss_fn_oe = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([1,10,10]).float())
        self.pesuo_loss_fn_sc = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([1,10,10,10]).float())

        self.my_min = my_min
        self.my_max = my_max
        
        self.sentence_encoder = sentence_encoder
        self.encoder_hidden_size = sentence_encoder.my_model.config.hidden_size
        self.my_classifier = My_classifier_model(self.encoder_hidden_size)
       
        
    def get_loss(self, logits_ae, logits_oe, logits_sc, my_target, pesuo_flag):
        if pesuo_flag:
            loss_ae = torch.mean(self.pesuo_loss_fn_ae(logits_ae, my_target["target_loss_labels"]), dim=-1) #(batch_size, sent_len)
            loss_oe = torch.mean(self.pesuo_loss_fn_oe(logits_oe, my_target["opinion_loss_labels"]), dim=-1)
            loss_sc = torch.mean(self.pesuo_loss_fn_sc(logits_sc, my_target["polarity_loss_labels"]), dim=-1)
        else:
            loss_ae = self.loss_fn_ae(logits_ae.permute(0,2,1), torch.argmax(my_target["target_loss_labels"], dim=-1))  #(batch_size, sent_len)
            loss_oe = self.loss_fn_oe(logits_oe.permute(0,2,1), torch.argmax(my_target["opinion_loss_labels"], dim=-1))
            loss_sc = self.loss_fn_sc(logits_sc.permute(0,2,1), torch.argmax(my_target["polarity_loss_labels"], dim=-1))

        return loss_ae, loss_oe, loss_sc
    
    def forward(self, inputs, my_target=None, if_get_loss=True, pesuo_flag=False, params=None):
        
        logits = self.sentence_encoder(inputs, params=self.get_subdict(params, 'sentence_encoder'))
        logits_ae, logits_oe, logits_sc =  self.my_classifier(logits, logits, logits, params=self.get_subdict(params, 'my_classifier'))
        
        if if_get_loss:
            loss_target, loss_opinion, loss_polarity = self.get_loss(logits_ae, logits_oe, logits_sc, my_target, pesuo_flag)
            total_loss = torch.mean(loss_target + loss_opinion + loss_polarity)
            
            dic_res = {}
            dic_res["logits_encoder"]=logits
            dic_res["logits_ae"]=logits_ae
            dic_res["logits_oe"]=logits_oe
            dic_res["logits_sc"]=logits_sc
            
            dic_res["total_loss"]=total_loss
            dic_res["loss_ae"]=loss_target
            dic_res["loss_oe"]=loss_opinion
            dic_res["loss_sc"]=loss_polarity
            return dic_res
        
        else:
            return logits_ae, logits_oe, logits_sc
            
   