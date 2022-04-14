import lxml.etree
doc = lxml.etree.parse('./ABSA-15_Laptops_Train_Data.xml')


def process(sent):
    while sent[-1]==".":
        sent = sent[:-1]
    while sent[-1]=="!":
        sent = sent[:-1]
    return sent.strip()



sent_list = []
ae_list = []
for sent in doc.xpath('/Reviews/Review/sentences/sentence/text'):
    p_sent = process(sent.text)
    sent_split = p_sent.split()
    
    # if sent_split[0]=="I" and sent_split[1]=="Contacted":
    #     print(1)
    
    sent_list.append(sent_split)
    lent = len(sent_split)
    ae_list.append( " ".join((["0"] * lent)) )
    

with open("train/sentence.txt", "w") as f:
    for i in sent_list:
        f.write(" ".join(i)+"\n")
        
with open("train/target.txt", "w") as f:
    for i in ae_list:
        f.write(i+"\n")
        
with open("train/opinion.txt", "w") as f:
    for i in ae_list:
        f.write(i+"\n")
        
with open("train/target_polarity.txt", "w") as f:
    for i in ae_list:
        f.write(i+"\n")