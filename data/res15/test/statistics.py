from collections import Counter
from transformers import BertTokenizer

with open("/export/home/hk52025804/workshop_NTU/SA/data/res15/test/sentence.txt", "r") as f:
    sent_data = f.readlines() 
with open("/export/home/hk52025804/workshop_NTU/SA/data/res15/test/target.txt", "r") as f:
    ae_data = f.readlines() 
with open("/export/home/hk52025804/workshop_NTU/SA/data/res15/test/opinion.txt", "r") as f:
    oe_data = f.readlines() 
with open("/export/home/hk52025804/workshop_NTU/SA/data/res15/test/target_polarity.txt", "r") as f:
    sc_data = f.readlines() 

tokenizer = BertTokenizer.from_pretrained("/export/home/hk52025804/workshop_NTU/SA/pretrain/bert-large-uncased")



def convert_BIO(ae_labels, oe_labels, sa_labels, raw_tokens, tokenizer):
    split_tokens = []
    split_ae_labels = []
    split_oe_labels = []
    split_sa_labels = []
    for ix, raw_token in enumerate(raw_tokens):
        raw_token = raw_token.lower()
        sub_tokens= tokenizer.tokenize(raw_token)
        for jx, sub_token in enumerate(sub_tokens):
            split_tokens.append(sub_token)
            if ae_labels[ix]==1 and jx>0:
                split_ae_labels.append(2)
            else:
                split_ae_labels.append(int(ae_labels[ix]))

            if oe_labels[ix]==1 and jx>0:
                split_oe_labels.append(2)
            else:
                split_oe_labels.append(int(oe_labels[ix]))
                
            if ix != len(raw_tokens)-1:   
                split_sa_labels.append(int(sa_labels[ix]))
            else:
                if len(sub_tokens)>1:
                    if ae_labels[-1] ==0 and jx<len(sub_tokens)-1:
                        split_sa_labels.append(0)
                    else:
                        split_sa_labels.append(int(sa_labels[ix]))
                else:
                    split_sa_labels.append(int(sa_labels[ix]))

    split_sa_labels_loss = []
    for l in split_sa_labels:
        # 0=background, 1=positive, 2=negative, 3=neutral, 4=conflict.
        if l == 0:
            split_sa_labels_loss.append(0)
        elif l == 1:
            split_sa_labels_loss.append(1)
        elif l == 2:
            split_sa_labels_loss.append(2)
        elif l == 3:
            split_sa_labels_loss.append(3)
        elif l == 4:
            split_sa_labels_loss.append(0)
        else:
            raise ValueError

    return split_tokens, split_ae_labels, split_oe_labels, split_sa_labels



count=0
total_ae = []
total_oe = []
total_sc = []
total_tag = []
total_tokens = []
total_sent = []
for sent, ae, oe, sc in zip(sent_data, ae_data, oe_data, sc_data):
    raw_tokens = sent.strip().split(" ")
    ae_labels = ae.strip().split(" ")
    oe_labels = oe.strip().split(" ")
    sa_labels = sc.strip().split(" ")
    
    total_sent.append(raw_tokens)
    total_tokens.extend(raw_tokens)

    total_ae.extend(ae_labels)
    total_oe.extend(oe_labels)
    total_sc.extend(sa_labels)


    split_tokens, split_ae_labels, split_oe_labels, split_sa_labels = convert_BIO(ae_labels, oe_labels, sa_labels, raw_tokens, tokenizer)
    total_tag.extend(split_ae_labels)
    total_tag.extend(split_oe_labels)
    total_tag.extend(split_sa_labels)

print("total_sent", len(total_sent))
print("total_tokens", len(total_tokens))
ae_dic = dict(Counter(total_ae))

sum_ae_O = ae_dic["0"]/len(total_tokens)
sum_ae_B = ae_dic["1"]/len(total_tokens)
sum_ae_I = ae_dic["2"]/len(total_tokens)

oe_dic = dict(Counter(total_oe))

sum_oe_O = oe_dic["0"]/len(total_tokens)
sum_oe_B = oe_dic["1"]/len(total_tokens)
sum_oe_I = oe_dic["2"]/len(total_tokens)

sc_dic = dict(Counter(total_sc))

# 1=positive, 2=negative, 3=neutral, 4=conflict.
sum_sc_pos = sc_dic["1"]/len(total_tokens)
sum_sc_neg = sc_dic["2"]/len(total_tokens)
sum_sc_neu = sc_dic["3"]/len(total_tokens)
sum_sc_other = sc_dic["0"]/len(total_tokens)


ae_token = round((ae_dic["1"]+ae_dic["2"]) /len(total_tokens)*100,2)
print("ae_token", ae_token)

oe_token = round((oe_dic["1"]+oe_dic["2"]) /len(total_tokens)*100,2)
print("oe_token", oe_token)


print("ae_dic", ae_dic)
print("ae", round(sum_ae_B*100,2), round(sum_ae_I*100,2), round(sum_ae_O*100,2))
print(round(round(sum_ae_B*100,2)+ round(sum_ae_I*100,2)+round(sum_ae_O*100,2),2))

print("oe_dic", oe_dic)
print("oe", round(sum_oe_B*100,2), round(sum_oe_I*100,2), round(sum_oe_O*100,2))
print(round(round(sum_oe_B*100,2)+ round(sum_oe_I*100,2)+round(sum_oe_O*100,2),2))

print("sc_dic", sc_dic)
print("sc", round(sum_sc_pos*100,2), round(sum_sc_neg*100,2), round(sum_sc_neu*100,2), round(sum_sc_other*100,2))
print(round(round(sum_sc_pos*100,2)+ round(sum_sc_neg*100,2)+round(sum_sc_neu*100,2)+round(sum_sc_other*100,2),2))

