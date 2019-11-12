import json
import pandas
import xmltodict

# g = open("./btrain.jsonl","w")
# h = open("./bval.jsonl","w")
g = open("./btest.jsonl","w")
with open("./copa-test.xml","r",encoding="utf-8") as f:
    lines = f.read()
    xmlparse = xmltodict.parse(lines)
    dic = xmlparse["copa-corpus"]
    dic2 = dic["item"]
    for i,line in enumerate(dic2):
        l={}
        l["premise"] = line["p"]
        l["choice1"] = line["a1"]
        l["choice2"] = line["a2"]
        l["question"] = line["@asks-for"]
        l["label"] = int(line["@most-plausible-alternative"])-1
        l["idx"] = int(i)
        g.write(json.dumps(l) + "\n")
