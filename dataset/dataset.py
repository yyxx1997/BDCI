import json
import os
from torch.utils.data import Dataset
from dataset.utils import pre_caption


class normal_dataset(Dataset):
    def __init__(self, ann_file, max_words=512):
        full_data = json.load(open(ann_file, 'r'))
        self.ann = []
        self.id2data = {}
        self.max_words = max_words
        order = 0
        for data in full_data:
            data['order'] = order
            self.ann.append(data)
            self.id2data[order] = data
            order += 1
        ...

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        title = ann['title']
        # assignee = ann['assignee']
        abstract = ann['abstract']
        label_id = int(ann['label_id']) if 'label_id' in ann.keys() else -1
        order = int(ann['order'])

        assemble_text = "这份专利的标题为：《{}》，详细说明如下：{}".format(
            title, abstract)[:self.max_words]

        return assemble_text, order, label_id
