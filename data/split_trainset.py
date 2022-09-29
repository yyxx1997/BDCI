import json
import random

trainset = []
with open('data/train.json','r',encoding="utf8") as file:
    for line in file:
        line = line.strip()
        dct = json.loads(line)
        trainset.append(dct)

random.shuffle(trainset)
length = len(trainset)
train_start, train_off = 0, int(length*0.85)
testset = trainset[train_off:]
trainset = trainset[train_start:train_off]
...

with open("data/train_{}.json".format(len(trainset)),'w+',encoding="utf8") as file:
    file.write(json.dumps(trainset,ensure_ascii=False,indent=4))

with open("data/test_{}.json".format(len(testset)),'w+',encoding="utf8") as file:
    file.write(json.dumps(testset,ensure_ascii=False,indent=4))

trainset = []
with open('data/testA.json','r',encoding="utf8") as file:
    for line in file:
        line = line.strip()
        dct = json.loads(line)
        trainset.append(dct)

with open("data/testA_format.json",'w+',encoding="utf8") as file:
    file.write(json.dumps(trainset,ensure_ascii=False,indent=4))