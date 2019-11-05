import json
import pandas


with open("./train.jsonl","r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = json.loads(line)
        l1 = len(tmp["premise"].split(" "))
        l2 = len(tmp["choice1"].split(" "))
        l3 = len(tmp["choice2"].split(" "))
        print(l1,l2,l3)
