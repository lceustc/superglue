import pandas as pd
from pandas import  DataFrame
from sklearn.model_selection import StratifiedKFold
import json
import os
# train=pd.read_csv('train1.tsv', sep='\t',header=0)
# data1 = DataFrame(train)
# print(data1)


# skf = StratifiedKFold(n_splits=5)
#
#
# for train_index, test_index in skf.split(data1[["premise","hypothesis"]],data1['label']):
#     print("TRAIN:", train_index, "TEST:", test_index)
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
file_list = [
             "./RTE-eda_9-alpha_.1-all",
             "./RTE-eda_9-alpha_.1-premise",
             "./RTE-eda_9-alpha_.2-all",
             "./RTE-eda_9-alpha_.2-premise",
             "./RTE-synonym_replacement_2-alpha_.1-all",
             "./RTE-synonym_replacement_2-alpha_.1-premise",
             "./RTE-synonym_replacement_2-alpha_.2-all",
             "./RTE-synonym_replacement_2-alpha_.2-premise"
             ]


g = open("./train02.tsv","a")
for i in file_list:
    if "2" in i.split(".")[-1]:
        with open(os.path.join(i,"train1.tsv"),"r") as f:
            tmp = f.readlines()
            for i in tmp:
                g.write(i)
g.close()


# max_len=0
# total=0
# num_256 = 0
# num_128 = 0
# for i in file_list:
#     if "1" in i :
#         with open(os.path.join(i,"train_1.jsonl"),"r") as f:
#             tmp = f.readlines()
#             for i in range(len(tmp)):
#                 total+=1
#                 data = json.loads(tmp[i])
#                 len_t = len(data['premise'].split(" "))+len(data['hypothesis'].split(" "))
#                 if len_t >256:
#                     num_256 +=1
#                 if len_t >max_len:
#                     max_len = len_t
#                 if len_t >128 :
#                     num_128 +=1

# with open("train.jsonl","r") as f:
#     tmp = f.readlines()
#     for i in range(len(tmp)):
#         total+=1
#         data = json.loads(tmp[i])
#         len_t = len(data['premise'].split(" "))+len(data['hypothesis'].split(" "))
#         if len_t >max_len:
#             max_len = len_t
#         if len_t >256:
#             num_256 +=1
#         if len_t >128 :
#             num_128 +=1
#
# with open("val.jsonl", "r") as f:
#     tmp = f.readlines()
#     for i in range(len(tmp)):
#         total+=1
#         data = json.loads(tmp[i])
#         len_t = len(data['premise'].split(" ")) + len(data['hypothesis'].split(" "))
#         if len_t > max_len:
#             max_len = len_t
#         if len_t>256 :
#             num_256+=1
#         if len_t > 128:
#             num_128 += 1
# print(total)
# print(num_128)
# print(num_256)
# print(max_len)

# g = open("./train2.tsv",'a')
# with open("train.jsonl",'r') as f:
#     tmp = f.readlines()
#     for i in range(len(tmp)):
#         data = json.loads(tmp[i])
#         g.writelines(str(data['premise'])+'\t'+str(data['hypothesis'])+'\t'+str(data['label']+'\t'+str(data['idx']) + '\n'))
# with open("val.jsonl",'r') as h :
#     tmpe = h.readlines()
#     for i in range(len(tmpe)):
#         data = json.loads(tmpe[i])
#         data['idx'] += 2490
#         g.writelines(str(data['premise'])+'\t'+str(data['hypothesis'])+'\t'+str(data['label']+'\t'+str(data['idx'])+'\n'))
# g.close()
# g = open("train2.tsv","a")
# for i in file_list:
#     with open(os.path.join(i,"train1.tsv"),"r") as f:
#         tmp = f.readlines()
#         for i in tmp:
#             g.write(i)
#
# g.close()
