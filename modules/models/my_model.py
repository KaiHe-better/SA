import torch
import torch.nn as nn
# from modules.models.feature_weight_module import Feature_weight_module
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaLinear
 
                           
class My_weighter(MetaModule):
    def __init__(self, max_len, batch_size, total_epoch):
        super(My_weighter, self).__init__()
        self.batch_size, self.max_len = batch_size, max_len
        
        label_embedding= 10
        epoch_embedding= 10
        loss_window =20
        self.loss_window =loss_window
        self.total_epoch =total_epoch
        
        self.ae_label_embedding = nn.Embedding(3, label_embedding)
        self.oe_label_embedding = nn.Embedding(3, label_embedding)
        self.sc_label_embedding = nn.Embedding(4, label_embedding)
        self.epoch_embedding = nn.Embedding(100, epoch_embedding)
        
        ae_input_dim = label_embedding+epoch_embedding+loss_window+3
        oe_input_dim = label_embedding+epoch_embedding+loss_window+3
        sc_input_dim = label_embedding+epoch_embedding+loss_window+4
        
        inner_div = 5
        
        self.fc1_ae = MetaLinear(ae_input_dim, int(ae_input_dim/inner_div))
        self.fc2_ae = MetaLinear(int(ae_input_dim/inner_div), 1)
        
        self.fc1_oe = MetaLinear(oe_input_dim, int(oe_input_dim/inner_div))
        self.fc2_oe = MetaLinear(int(oe_input_dim/inner_div), 1)
        
        self.fc1_sc = MetaLinear(sc_input_dim, int(sc_input_dim/inner_div))
        self.fc2_sc = MetaLinear(int(sc_input_dim/inner_div), 1)
        
    def normalize_epoch(self, epoch):
        return int(((float(epoch) - 0)/(self.total_epoch - 0))*100) 
        
    def forward(self, res_dic, my_target, epoch, params=None):
        loss_target = res_dic["loss_ae"].unsqueeze(-1).repeat(1,1,self.loss_window)
        loss_opinion = res_dic["loss_oe"].unsqueeze(-1).repeat(1,1,self.loss_window)
        loss_polarity = res_dic["loss_sc"].unsqueeze(-1).repeat(1,1,self.loss_window)
        
        logit_target, logit_opinion, logit_polarity = res_dic["logits_ae"], res_dic["logits_oe"], res_dic["logits_sc"]
        label_target, label_opinion, label_polarity = my_target["target_loss_labels"], my_target["opinion_loss_labels"], my_target["polarity_loss_labels"]
        
        epoch = self.epoch_embedding(torch.tensor(self.normalize_epoch(epoch)).to(loss_target.device)).repeat(loss_target.size()[0], self.max_len, 1)
        label_embeds_ae = self.ae_label_embedding(torch.argmax(label_target, dim=-1))
        label_embeds_oe = self.oe_label_embedding(torch.argmax(label_opinion, dim=-1))
        label_embeds_sc = self.sc_label_embedding(torch.argmax(label_polarity, dim=-1))
        
        input_embeds_ae =  torch.cat((loss_target, logit_target, label_embeds_ae, epoch), dim=-1)
        input_embeds_oe =  torch.cat((loss_opinion, logit_opinion, label_embeds_oe, epoch), dim=-1)
        input_embeds_sc =  torch.cat((loss_polarity, logit_polarity, label_embeds_sc, epoch), dim=-1)
        
        total_loss_weight = self.weight_loss_layer(input_embeds_ae, input_embeds_oe, input_embeds_sc, params=self.get_subdict(params, 'weight_loss_layer'))
        weight_loss = self.get_weighted_toal_loss(total_loss_weight, res_dic)
        
        return weight_loss, total_loss_weight
    
    def weight_loss_layer(self, input_embeds_ae, input_embeds_oe, input_embeds_sc, params=None):
        losits_ae = self.fc1_ae(input_embeds_ae, params=self.get_subdict(params, 'fc1_ae'))
        losits_ae = F.relu(losits_ae)
        losits_ae = self.fc2_ae(losits_ae, params=self.get_subdict(params, 'fc2_ae'))
        losits_ae = F.sigmoid(losits_ae)
        
        losits_oe = self.fc1_oe(input_embeds_oe, params=self.get_subdict(params, 'fc1_oe'))
        losits_oe = F.relu(losits_oe)
        losits_oe = self.fc2_oe(losits_oe, params=self.get_subdict(params, 'fc2_oe'))
        losits_oe = F.sigmoid(losits_oe)
        
        losits_sc = self.fc1_sc(input_embeds_sc, params=self.get_subdict(params, 'fc1_sc'))
        losits_sc = F.relu(losits_sc)
        losits_sc = self.fc2_sc(losits_sc, params=self.get_subdict(params, 'fc2_sc'))
        losits_sc = F.sigmoid(losits_sc)
        
        return [losits_ae, losits_oe, losits_sc]
    
    def get_weighted_toal_loss(self, weights, res_dic):
        loss_ae = torch.mean(weights[0].squeeze() * res_dic["loss_ae"]) # weights = (batch_size, 3) res_dic["loss_ae"] = (batch_size, sent_len)
        loss_oe = torch.mean(weights[1].squeeze() * res_dic["loss_oe"])
        loss_sc = torch.mean(weights[2].squeeze() * res_dic["loss_sc"])
        
        # loss_ae = torch.mean(weights[0]) * torch.mean(res_dic["loss_ae"]) 
        # loss_oe = torch.mean(weights[1]) * torch.mean(res_dic["loss_oe"])
        # loss_sc = torch.mean(weights[2]) * torch.mean(res_dic["loss_sc"])
        
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
    def __init__(self, sentence_encoder, if_weight_loss, max_length, my_min=0.9, my_max=3, non_o_weig=10):
        super(My_model, self).__init__()
        self.loss_fn_ae = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,non_o_weig,non_o_weig]).float())
        self.loss_fn_oe = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,non_o_weig,non_o_weig]).float())
        self.loss_fn_sc = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1,non_o_weig,non_o_weig,non_o_weig]).float())
        
        self.pesuo_loss_fn_ae = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([1,non_o_weig,non_o_weig]).float())
        self.pesuo_loss_fn_oe = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([1,non_o_weig,non_o_weig]).float())
        self.pesuo_loss_fn_sc = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([1,non_o_weig,non_o_weig,non_o_weig]).float())

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
            
   