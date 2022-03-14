import os
data_set_list = ["res14", "res15", "lap14"]
temp = ["train", "dev", "test"]

def pam_new_tag(index_list, data_index, a, o, s):
    
    each_sent_tag_list = []
    for a_item, o_item, s_item in zip(a.strip().split(), o.strip().split(), s.strip().split()) :
        if a_item != "0":
            if a_item ==o_item:
                index_list.append(data_index)
        a_item = int(a_item)
        o_item = int(o_item)
        s_item = int(s_item)
                
        # 0=O, 1=B, 2=I.   
        if a_item == 0:
            temp_a_tag= "O"
        elif a_item == 1:
            temp_a_tag= "AB"
        elif a_item == 2:    
            temp_a_tag= "AI"
        else:
            raise Exception("error")

                # 0=O, 1=B, 2=I.
        if o_item == 0:
            temp_o_tag= "O"
        else:
            if a_item != 0:
                temp_o_tag= "O"
            else:
                if o_item == 1:
                    temp_o_tag= "OB"
                elif o_item == 2:    
                    temp_o_tag= "OI"
                else:
                    raise Exception("error")
                
        # 0=background, 1=positive, 2=negative, 3=neutral, 4=conflict.
        if s_item == 0:
            temp_s_tag= "O"
        elif s_item == 1:
            temp_s_tag= "pos"
        elif s_item == 2:    
            temp_s_tag= "neg"
        elif s_item == 3:    
            temp_s_tag= "neu"
        elif s_item == 4:    
            temp_s_tag= "O"
        else:
            raise Exception("error")
                
        new_tag = temp_a_tag + "_" + temp_o_tag+ "_"+temp_s_tag
        each_sent_tag_list.append(new_tag)
    return each_sent_tag_list


tag_to_index = []
for dataset in data_set_list:
    for data in temp:
        file_target = os.path.join(dataset, os.path.join(data, "target.txt"))
        file_opinion = os.path.join(dataset, os.path.join(data, "opinion.txt"))
        file_target_polarity = os.path.join(dataset, os.path.join(data, "target_polarity.txt"))

        with open(file_target, "r") as f:
            a_data=f.readlines()

        with open(file_opinion, "r") as f:
            o_data=f.readlines()
        
        with open(file_target_polarity, "r") as f:
            s_data=f.readlines()  
            
        index_list = []
        combined_tag_list = []
        for data_index, (a, o, s) in enumerate(zip(a_data, o_data, s_data)):
            
            each_sent_tag_list = pam_new_tag(index_list, data_index, a, o, s)
            combined_tag_list.append(each_sent_tag_list)
            tag_to_index.extend(each_sent_tag_list)
        print(file_target)
        print(len(index_list))
        print(index_list)
        print("---------------------")
    
tag_to_index_dic = {}     
tag_to_index = list(set(tag_to_index))
for i in range(len(tag_to_index)):
    tag_to_index_dic[tag_to_index[i]] = i
print(tag_to_index_dic)   