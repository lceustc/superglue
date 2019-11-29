import json
import spacy
#
#
nlp = spacy.load("en_core_web_sm")
g = open("./val1.jsonl","a")



with open("./val.jsonl","r") as f:
    lines = f.readlines()
    for line in lines:
        dic = json.loads(line)
        sen1 = dic["sentence1"]
        sen2 = dic["sentence2"]
        doc1 = nlp(sen1)
        doc2 = nlp(sen2)

        word1 = sen1[dic["start1"]:(dic["end1"])]
        word2 = sen2[dic["start2"]:(dic["end2"])]
        pos1,pos2 = None,None
        for token in doc1:
            if token.text == word1:
                pos1 = str(token.pos_)
        for token in doc2:
            if token.text == word2:
                pos2 = str(token.pos_)
        if pos1 is None or pos2 is None:
            print("word doesn't match sen1 or sen2")
            print(word1,word2)
            print(sen1,sen2)
            print("--------------------")

        dic["pos1"] = pos1
        dic["pos2"] = pos2
        g.write(json.dumps(dic))
        g.write("\n")





# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
#
# for token in doc:
#     # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#     #         token.shape_, token.is_alpha, token.is_stop)
#     print(token.text)
#     print(token.text == "Apple")
#     break