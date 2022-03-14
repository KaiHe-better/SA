import lxml.etree
doc = lxml.etree.parse('./ABSA-15_Laptops_Train_Data.xml')

sent_list = []
ae_list = []
for sent in doc.xpath('/Reviews/Review/sentences/sentence/text'):
    sent_list.append(sent.text)
    lent = len(sent.text.split(" "))
    ae_list.append( " ".join(["0"] * lent) )

with open("train/sentence.txt", "w") as f:
    for i in sent_list:
        f.write(i+"\n")
        
with open("train/target.txt", "w") as f:
    for i in ae_list:
        f.write(i+"\n")
        
with open("train/opinion.txt", "w") as f:
    for i in ae_list:
        f.write(i+"\n")
        
with open("train/target_polarity.txt", "w") as f:
    for i in ae_list:
        f.write(i+"\n")