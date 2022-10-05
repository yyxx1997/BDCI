import json
import random
import torch
import webob

category = {}
with open('train.json','r',encoding="utf8") as file:
    for line in file:
        line = line.strip()
        dct = json.loads(line)
        cate = dct['label_id']
        if cate in category.keys():
            category[cate]['number'] += 1
            category[cate]['content'].append(dct)
        else:
            category[cate] = {'number':0, 'content':[]}

total_sample = sum([i['number'] for i in category.values()])
weights = torch.Tensor([i['number']/total_sample for i in category.values()])
weights = weights/torch.max(weights)
print(weights.tolist(), file=open("weights.txt",'w'))
category = dict(sorted(category.items(),key=lambda x:x[1]['number'],reverse=True))

with open("trainset_classified.json",'w',encoding='utf8') as file:
    file.write(json.dumps(category,ensure_ascii=False,indent=4))
