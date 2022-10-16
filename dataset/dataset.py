import json
import os
import torch
from torch.utils.data import Dataset
from dataset.utils import pre_caption


class normal_dataset(Dataset):
    def __init__(self, ann_file, tokenizer, max_words=512):
        full_data = json.load(open(ann_file, 'r'))
        self.ann = []
        self.id2data = {}
        self.max_words = max_words
        self.tokenizer = tokenizer
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

        return (assemble_text, order, label_id)

    def collate_fn(self, inputs):
        # update final outputs
        final_assemble_text = []
        final_order = []
        final_label_id = []

        for input in inputs:
            assemble_text, order, label_id = input
            final_assemble_text.append(assemble_text)
            final_order.append(order)
            final_label_id.append(label_id)
        
        final_assemble_text = self.tokenizer(final_assemble_text, padding='longest', return_tensors="pt")
        final_order = torch.tensor(final_order)
        final_label_id = torch.tensor(final_label_id)

        batch={}
        batch['text'] = final_assemble_text
        batch['order'] = final_order
        batch['label'] = final_label_id
        return batch