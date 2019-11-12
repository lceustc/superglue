import json


g = open("./train_all.tsv","w")
with open("train_all.jsonl","r") as f:
    t = f.readlines()
    for i in t:
        data = json.loads(i)
        g.writelines(str(data['premise']) + '\t' + str(data['hypothesis']) + '\t' + str(
            data['label'] + '\t' + str(data['idx']) + '\n'))