import torch 
import os


def convert_to_list(y_aspect, y_sentiment, mask):
    y_aspect_list = []
    y_sentiment_list = []
    for seq_aspect, seq_sentiment, seq_mask in zip(y_aspect, y_sentiment, mask):
        l_a = []
        l_s = []
        for label_dist_a, label_dist_s, m in zip(seq_aspect, seq_sentiment, seq_mask):
            if m == 0:
                break
            else:
                if label_dist_a.view(-1).size()[-1]==1:
                    l_a.append(int(label_dist_a))
                    l_s.append(int(label_dist_s))
                else:
                    l_a.append(int(torch.argmax(label_dist_a)))
                    l_s.append(int(torch.argmax(label_dist_s)))
        y_aspect_list.append(l_a)
        y_sentiment_list.append(l_s)
    return y_aspect_list, y_sentiment_list

def score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, train_op, index_list, token_list):
    if train_op:
        begin = 1
        inside = 2
    else:
        begin = 1
        inside = 2

        # predicted sentiment distribution for aspect terms that are correctly extracted
        pred_count = {"background_conflict":0, 'pos':0, 'neg':0, 'neu':0}
        
        # gold sentiment distribution for aspect terms that are correctly extracted
        rel_count = {"background_conflict":0, 'pos':0, 'neg':0, 'neu':0}
        
        # sentiment distribution for terms that get both span and sentiment predicted correctly
        correct_count = {"background_conflict":0, 'pos':0, 'neg':0, 'neu':0}
        
        # sentiment distribution in original data
        total_count = {"background_conflict":0, 'pos':0, 'neg':0, 'neu':0}

        polarity_map = {0:"background_conflict", 1: 'pos', 2: 'neg', 3: 'neu'}

        # count of predicted conflict aspect term
        predicted_conf = 0

    correct, predicted, relevant = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq = true_aspect[i]
        predict = predict_aspect[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == begin:
                relevant += 1
                if not train_op:
                    if true_sentiment[i][num]!=0:
                        total_count[polarity_map[true_sentiment[i][num]]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside  and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1
                        if not train_op:
                            # do not count conflict examples
                            if true_sentiment[i][num]!=0:
                                rel_count[polarity_map[true_sentiment[i][num]]]+=1
                                pred_count[polarity_map[predict_sentiment[i][num]]]+=1
                                if true_sentiment[i][num] == predict_sentiment[i][num]:
                                    correct_count[polarity_map[true_sentiment[i][num]]]+=1
                            else:
                                predicted_conf += 1

        for pred in predict:
            if pred == begin:
                predicted += 1

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    # F1 score for aspect (opinion) extraction
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

    acc_s, f_s, f_absa = 0, 0, 0
    pr_s, re_s = 0, 0
    precision_absa, recall_absa = 0, 0
    if not train_op:
        num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
        num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
        num_total = total_count['pos']+total_count['neg']+total_count['neu']

        acc_s = num_correct_overall/(num_correct_aspect+1e-6)
       
        p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
        r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
        
        p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
        r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

        p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
        r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

        pr_s = (p_pos+p_neg+p_neu)/3.0
        re_s = (r_pos+r_neg+r_neu)/3.0

        # For calculating the F1 Score for SC, we have discussed with Ruidan at https://github.com/ruidan/IMN-E2E-ABSA/issues?q=is%3Aissue+is%3Aclosed.
        # We provide the correct formula as follow, but we still adopt the calculation in IMN to conduct a fair comparison.
        # f_pos = 2*p_pos*r_pos /(p_pos+r_pos+1e-6)
        # f_neg = 2*p_neg*r_neg /(p_neg+r_neg+1e-6)
        # f_neu = 2*p_neu*r_neu /(p_neu+r_neu+1e-6)
        # f_s_1 = (f_pos+f_neg+f_neu)/3.0
        # print("f_s_1", f_s_1)

        # F1 score for SC only (in IMN)
        f_s = 2*pr_s*re_s/(pr_s+re_s+1e-6)

        precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
        recall_absa = num_correct_overall/(num_total+1e-6)
        f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)
        
    return (p_aspect, r_aspect, f_aspect), acc_s, (pr_s, re_s, f_s), (precision_absa, recall_absa, f_absa)

def record_res(train_flag, res_path, index_list, token_list, 
               true_aspect, true_sentiment, true_opinion, 
               predict_aspect, predict_sentiment, predict_opinion, record_flag, res_list):
    if res_list is not None:
        gold_ae_list = res_list[0]
        gold_oe_list = res_list[1]
        gold_sc_list = res_list[2]
    else:
        gold_ae_list = true_aspect
        gold_oe_list = true_opinion
        gold_sc_list = true_sentiment
        
    with open(os.path.join(res_path, train_flag+"_"+str(record_flag)+"_res.txt"), "w") as f:
        for index, token, item_true_aspect, item_true_sentiment, item_true_opinion, \
            item_predict_aspect, item_predict_sentiment, item_predict_opinion, \
            gold_ae, gold_oe, gold_sc in \
                zip(index_list, token_list, true_aspect, true_sentiment, true_opinion, 
                    predict_aspect, predict_sentiment, predict_opinion, 
                    gold_ae_list, gold_oe_list, gold_sc_list):
            
            my_len = len(item_true_opinion)
            f.write("index: "+str(index)+"\n")                                              
            f.write("token: "+str(token)+"\n")
            f.write("\n")
            if res_list is not None:                                        
                f.write("glod_aspect:    "+str(gold_ae.tolist()[:my_len])+"\n")                                                
            f.write("pesuo_aspect:   "+str(item_true_aspect)+"\n")                                                
            f.write("predict_aspect: "+str(item_predict_aspect)+"\n")  
            f.write("\n")   
            if res_list is not None:                                        
                f.write("glod_opinion:    "+str(gold_oe.tolist()[:my_len])+"\n")                                          
            f.write("pesuo_opinion:   "+str(item_true_opinion)+"\n")                                             
            f.write("predict_opinion: "+str(item_predict_opinion)+"\n")   
            f.write("\n")   
            if res_list is not None:                                        
                f.write("glod_sentiment:    "+str(gold_sc.tolist()[:my_len])+"\n")                                        
            f.write("pesuo_sentiment:   "+str(item_true_sentiment)+"\n")                                             
            f.write("predict_sentiment: "+str(item_predict_sentiment)+"\n")
                                                      
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")

def get_new_predict(batch_target_dic, y_predict_aspect):
    tag_to_index = batch_target_dic["tag_to_index"]
    aspect_to_tag_dic = batch_target_dic["aspect_to_tag_dic"]
    opinion_to_tag_dic = batch_target_dic["opinion_to_tag_dic"]
    sentiment_to_tag_dic = batch_target_dic["sentiment_to_tag_dic"]
    
    tag_to_index_new=dict(zip(tag_to_index.values(),tag_to_index.keys()))
    aspect_to_tag_dic=dict(zip(aspect_to_tag_dic.values(),aspect_to_tag_dic.keys()))
    opinion_to_tag_dic=dict(zip(opinion_to_tag_dic.values(),opinion_to_tag_dic.keys()))
    sentiment_to_tag_dic=dict(zip(sentiment_to_tag_dic.values(),sentiment_to_tag_dic.keys()))
    
    
    new_y_predict_aspect = []
    new_y_predict_opinion = []
    new_y_predict_sentiment = []
    for sent in y_predict_aspect:
        temp_y_predict_aspect = []
        temp_y_predict_opinion = []
        temp_y_predict_sentiment = []
        for item in sent:
            tag = tag_to_index_new[int(item)].split("_")
            temp_y_predict_aspect.append(aspect_to_tag_dic[tag[0]])
            temp_y_predict_opinion.append(opinion_to_tag_dic[tag[1]])
            temp_y_predict_sentiment.append(sentiment_to_tag_dic[tag[2]])
        new_y_predict_aspect.append(temp_y_predict_aspect)
        new_y_predict_opinion.append(temp_y_predict_opinion)
        new_y_predict_sentiment.append(temp_y_predict_sentiment)
    return new_y_predict_aspect, new_y_predict_sentiment, new_y_predict_opinion

def get_metric(train_flag, res_path, index_list, token_list, 
               y_true_aspect, y_true_opinion, y_true_sentiment,
               y_predict_aspect, y_predict_opinion, y_predict_sentiment, 
               mask, epoch=0,  record_flag=False, res_list=None, train_op=1):

    true_aspect, true_sentiment = convert_to_list(y_true_aspect, y_true_sentiment, mask)
    predict_aspect, predict_sentiment = convert_to_list(y_predict_aspect, y_predict_sentiment, mask)

    true_opinion, _ = convert_to_list(y_true_opinion, y_true_sentiment, mask)
    predict_opinion, _ = convert_to_list(y_predict_opinion, y_predict_sentiment, mask)

    prf_aspect, _, prf_s, prf_absa = score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, 0, index_list, token_list)
    
    if train_op:
        prf_opinion, _, _, _ = score(true_opinion, predict_opinion, true_sentiment, predict_sentiment, 1, index_list, token_list)
    
    # if record_flag and (res_list is not None):
    #     record_res(train_flag, res_path, index_list, token_list, 
    #             true_aspect, true_sentiment, true_opinion, 
    #             predict_aspect, predict_sentiment, predict_opinion, record_flag, res_list)

    return prf_aspect, prf_opinion, prf_s, prf_absa



