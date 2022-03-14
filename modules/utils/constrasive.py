import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, metric = 'l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)


    def forward(self, x0, x1, y):
        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod /  torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist
    
def contrastive_loss(class_output, LM_output, distmetric = 'l2'):
        # softmax = nn.Softmax(dim=1)
        # target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        # if conf == 'max':
        #     weight = torch.max(target, axis = 1).values
        #     w = torch.tensor([i for i,x in enumerate(weight) if x > thresh], dtype=torch.long).to(input.device)
        # elif conf == 'entropy':
        #     weight = torch.sum(-torch.log(target+1e-6) * target, dim = 1)
        #     weight = 1 - weight / np.log(weight.size(-1))
        #     w = torch.tensor([i for i,x in enumerate(weight) if x > thresh], dtype=torch.long).to(input.device)
        # input_x = input[w]
        # feat_x = LM_output[w]
        
        feat_x = LM_output.view(-1, LM_output.shape[-1])
        input_x = class_output.view(-1, class_output.shape[-1])
        batch_size = input_x.size()[0]
        if batch_size == 0:
            return 0
        index = torch.randperm(batch_size).to(feat_x.device)
        input_y = input_x[index, :]
        feat_y = feat_x[index, :]
        argmax_x = torch.argmax(input_x, dim = -1)
        argmax_y = torch.argmax(input_y, dim = -1)
        agreement = torch.FloatTensor([1 if x == True else 0 for x in argmax_x == argmax_y]).to(feat_x.device)

        criterion = ContrastiveLoss(margin = 1.0, metric = distmetric)
        loss, dist_sq, dist = criterion(feat_x, feat_y, agreement)
        
        return loss
    
def soft_frequency(logits,  probs=False, soft = True):
        power = 2
        if not probs:
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y**power / f
        #print('t', t)
        t = t + 1e-10
        p = t/torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)
    
def calc_loss(input, target, thresh = 0.95, soft = True, conf = 'max', confreg = 0.1):
    softmax = nn.Softmax(dim=1)
    target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
    
    if conf == 'max':
        weight = torch.max(target, axis = 1).values
        w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(target.device)
    elif conf == 'entropy':
        weight = torch.sum(-torch.log(target+1e-6) * target, dim = 1)
        weight = 1 - weight / np.log(weight.size(-1))
        w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(target.device)
        
    target = soft_frequency(target, probs = True, soft = soft)
    loss = nn.KLDivLoss(reduction = 'none') if soft else nn.CrossEntropyLoss(reduction = 'none')
    loss_batch = loss(input, target)

    l = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))
    
    n_classes_ = input.shape[-1]
    l -= confreg *( torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_ )
    return l
    
def R_drop_loss(logit_1, logit_2):
    kl_1 = F.kl_div(logit_1.softmax(dim=-1).log(), logit_2.softmax(dim=-1), reduction='sum')
    kl_2 = F.kl_div(logit_2.softmax(dim=-1).log(), logit_1.softmax(dim=-1), reduction='sum')
    return (kl_1+kl_2)/2


if __name__ == '__main__':
    my_input = torch.rand(32, 100, 3)
    my_feat = torch.rand(32, 100, 1024)
    
    # my_input = torch.rand(32, 3)
    # my_feat = torch.rand(32, 1024)
    
    my_target = torch.rand(32, 3)
    

    """ input = torch.log(softmax(logits))      student classifier output
    
    feat = outputs_pseudo["hidden_states"]  Teacher bert output
    target = outputs_pseudo["logits"]       Teacher classifier output, for soft pseudo label
    """
    c_loss = contrastive_loss(my_input, my_feat, my_target)
    print(c_loss)