import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ID', default='0', type=str, help='ID')
parser.add_argument('--GPU', default='3', type=str, help='GPU')
parser.add_argument('--corpus', default='lap14', type=str, choices=['res14', 'lap14', 'res15'], help='corpus')
parser.add_argument('--batch_size', default=64, type=int, help='number of example per batch :  64')
parser.add_argument('--optim', default='adamw', help='sgd / adam / adamw')
parser.add_argument('--max_len', default=100, type=int, help='max_len')
parser.add_argument('--augment_num', default=2, type=int, help='augment_num')
parser.add_argument('--epochs', default=1, type=int, help='inner epochs nums')
parser.add_argument('--inner_epoch', default=2, type=int, help='inner epochs nums')
parser.add_argument('--total_epochs', default=1000, type=int, help='total_epochs')

parser.add_argument('--break_flag', action='store_true', default=False,  help='weight_loss')
parser.add_argument('--if_weight_loss', action='store_true', default=False,  help='weight_loss')
parser.add_argument('--no_self_training', action='store_true', default=False,  help='add_other_corpus')

parser.add_argument('--add_other_corpus', action='store_true', default=False,  help='add_other_corpus')
parser.add_argument('--my_min', default=1, type=float, help='my_min')
parser.add_argument('--my_max', default=3, type=float, help='my_max')
parser.add_argument('--non_o_weig', default=30, type=float, help='my_max')

parser.add_argument('--never_end', action='store_true', default=False,  help='weight_loss')
parser.add_argument('--if_constra', action='store_true', default=False,  help='if_constra')
parser.add_argument('--contra_loss_weight', default=0.05, type=float, help='augment_num')

parser.add_argument('--only_test', action='store_true', default=False,  help='only test')
parser.add_argument('--if_fp16', action='store_true', default=True,  help='if_fp16')
parser.add_argument('--load_ckpt_path', default=None, type=str, help='path for loading an existing checkpoint')

parser.add_argument('--chosed_th', default=0.4, type=float, help='augment_num')
parser.add_argument('--uncertain_th_S', default=0.2, type=float, help='augment_num')
parser.add_argument('--uncertain_th_E', default=0.9, type=float, help='augment_num')

parser.add_argument('--data_ratio', default=1, type=float, help='augment_num')
parser.add_argument('--remain_pesudo_data_ratio', default=1, type=float, help='1 means 1-data_ratio unlabel data')
parser.add_argument('--mix_loss_weight', default=0.5, type=float, help='augment_num')

parser.add_argument('--warmup_step', default=100, type=int, help='learning rate')
parser.add_argument('--encoder_lr', default=4e-5, type=float, help='learning rate')
parser.add_argument('--weight_lr', default=8e-5, type=float, help='learning rate')
parser.add_argument('--classifier_lr', default=8e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.5, type=float, help='L2')
opt = parser.parse_args()


import os
import sys
import warnings
from numpy.random import seed
from .train_frame import My_Train_Framework
from .encoders.bart_encoder import BART_encoder
from .encoders.bert_encoder import BERT_encoder
from .encoders.roberta_encoder import Roberta_encoder
from .utils.my_utils import print_execute_time
from .models.my_model import My_model, My_weighter
from dataloader.dataloader import get_loader

CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.GPU
warnings.filterwarnings("ignore")

import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
print("opt.ID :", opt.ID)
print("batch_size :", opt.batch_size)
print("if_weight_loss :", opt.if_weight_loss)
print("if_constra :", opt.if_constra)
print("if_fp16 :", opt.if_fp16)
print("break_flag :", opt.break_flag)
print("\n")

if torch.cuda.is_available():
     print("using GPU {} ...".format(opt.GPU))
else:
    print("using CPU...")

SEED = 123
seed(SEED)  

if opt.remain_pesudo_data_ratio!=1:
    assert (opt.data_ratio + opt.remain_pesudo_data_ratio)<=1
prefix = str(sys.argv[1:])
opt.save_ckpt_path = 'checkpoint/{}'.format(prefix)

opt.load_ckpt_path = 'checkpoint/{}'.format(opt.load_ckpt_path)
# if opt.corpus =="res14":
#     opt.load_ckpt_path = 'checkpoint_saved/res14-73'
# if opt.corpus =="res15":
#     opt.load_ckpt_path = 'checkpoint_saved/res15-66'
# if opt.corpus =="lap14":
#     opt.load_ckpt_path = 'checkpoint_saved/lap14-65'

if opt.break_flag:
    opt.chosed_th= 0.36
    opt.warmup_step= 1
   
    
    
def get_model():
    my_weighter = My_weighter(opt.max_len, opt.batch_size, opt.total_epochs)
    
    tea_sentence_encoder = BERT_encoder("./pretrain/bert-large-uncased")
    tea_model = My_model(tea_sentence_encoder, opt.if_weight_loss, opt.max_len, opt.my_min, opt.my_max, opt.non_o_weig)
    
    stu_sentence_encoder = BERT_encoder("./pretrain/bert-large-uncased")
    stu_model = My_model(stu_sentence_encoder, opt.if_weight_loss, opt.max_len, opt.my_min, opt.my_max, opt.non_o_weig)
    
    return tea_model, stu_model, my_weighter



@print_execute_time
def main(train_index, train_ID=None):
    tea_model, stu_model, my_weighter = get_model()
    data_loader_list, train_dataset = get_loader(opt.corpus, opt.data_ratio, opt.remain_pesudo_data_ratio, opt.add_other_corpus, opt.batch_size, opt.max_len, tea_model.sentence_encoder.tokenizer)
    train_data_loader, dev_data_loader, test_data_loader, pesu_data_loader = data_loader_list
    my_train_framework = My_Train_Framework(train_ID, train_index, train_data_loader, dev_data_loader, test_data_loader, pesu_data_loader, train_dataset, opt)
        
    if not opt.no_self_training:
        best_epoch, best_dic_F = my_train_framework.train(tea_model, stu_model, my_weighter, opt.optim, opt.save_ckpt_path, opt.load_ckpt_path, 
                                total_epochs=opt.total_epochs, corpus=opt.corpus, if_weight_loss=opt.if_weight_loss, 
                                weight_decay=opt.weight_decay, encoder_lr=opt.encoder_lr, weight_lr=opt.weight_lr, classifier_lr=opt.classifier_lr)
    else:
        best_epoch, best_dic_F = my_train_framework.train_no_self_training(tea_model, stu_model, my_weighter, opt.optim, opt.save_ckpt_path, opt.load_ckpt_path, 
                                total_epochs=opt.total_epochs, corpus=opt.corpus, if_weight_loss=opt.if_weight_loss, 
                                weight_decay=opt.weight_decay, encoder_lr=opt.encoder_lr, weight_lr=opt.weight_lr, classifier_lr=opt.classifier_lr)
        
        
    # model, _, _ = my_train_framework.__initialize__(my_model, opt.load_ckpt_path, only_test=True)
    # _ = my_train_framework.eval(train_flag="Test", model=model, ) 
    return best_epoch, best_dic_F, opt