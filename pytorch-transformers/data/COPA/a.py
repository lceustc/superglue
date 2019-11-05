import json
import pandas

g = open("./train.tsv","a")

with open("./train.jsonl","r") as f:

    lines = f.readlines()
    for line in lines:
        tmp = json.loads(line)
        g.write(str(tmp["premise"])+"\t"+str(tmp["choice1"])+"\t"+str(tmp["choice2"])+"\t"+str(tmp["question"])+"\t"+str(tmp["label"])+"\t"+str(tmp["idx"])+"\n")

g.close()