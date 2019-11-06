import json
import pandas
import xmltodict

g = open("./btrain.jsonl","w")
h = open("./bval.jsonl","w")
with open("./balacopa-dev-all.xml","r",encoding="utf-8") as f:
    lines = f.read()
    xmlparse = xmltodict.parse(lines)
    dic = xmlparse["copa-corpus"]
    dic2 = dic["item"]
    for i,line in enumerate(dic2):
        # if int(line["@id"]) <600:
        #     continue
        l = {}
        l["premise"] = line["p"]
        l["choice1"] = line["a1"]
        l["choice2"] = line["a2"]
        l["question"] = line["@asks-for"]
        l["label"] = int(line["@most-plausible-alternative"])-1
        # if 0<=int(line["@id"])-1001<=399:
        l["idx"] = int(i)
        g.write(json.dumps(l) + "\n")
        # else:
        #     l["idx"] = int(line["@id"])-1401
        #     h.write(str(l)+"\n")
