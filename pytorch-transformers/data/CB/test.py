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
             "./CB-eda_9-alpha_.1-all",
             "./CB-eda_9-alpha_.1-premise",
             "./CB-eda_9-alpha_.2-all",
             "./CB-eda_9-alpha_.2-premise",
             "./CB-synonym_replacement_2-alpha_.1-all",
             "./CB-synonym_replacement_2-alpha_.1-premise",
             "./CB-synonym_replacement_2-alpha_.2-all",
             "./CB-synonym_replacement_2-alpha_.2-premise"
             ]
# for i in file_list:
#     with open(i,'r') as f:
#         tmp = f.readlines()
#     for i in range(len(tmp)):
#         data = json.loads(tmp[i])
#         print(type(data['idx']))
#         # data['idx'] = int(data['idx'])+1
#         # print(data['idx'])
#         break
# #     break
# for i in file_list:
#     t = os.path.join(i,"train.jsonl")
#     with open(t,'r') as f:
#         tmp = f.readlines()
#     op = os.path.join(i,"train.tsv")
#     g = open(op,'w')
#     for j in range(len(tmp)):
#         data = json.loads(tmp[j])
#         if data['idx']>249:
#             data['idx'] +=1
#         g.writelines(str(data['premise'])+'\t'+str(data['hypothesis'])+'\t'+str(data['label']+'\t'+str(data['idx'])+'\n'))

# tmp = pd.read_csv("./train2.tsv",sep='\t',header=0)
# # print(tmp.iloc[1901,[-1]])
# # tmp.iloc[1901,[-1]] = int(tmp.iloc[1901,[-1]])+1
# # print(tmp.iloc[1901,[-1]])
# # print(tmp.iloc[1900,[-1]])
# for i in range(1901,len(tmp)):
#     tmp.iloc[i,[-1]] = int(tmp.iloc[i,[-1]])+1
# # print(tmp['idx'])
# tmp = DataFrame(tmp)
# tmp.to_csv("./temp.tsv",sep='\t',header=False,index=False)
i= 0
with  open("./test.jsonl","r") as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        l1 = len(line["premise"].split(" "))
        l2 = len(line["hypothesis"].split(" "))
        if l1+l2 >123:
            i+=1
    print(1-(i/len(lines)))
