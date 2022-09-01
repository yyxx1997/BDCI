from functools import partial
from transformers import BertModel
import torch
from torch import nn
import torch.nn.functional as F


class BaselineBert(nn.Module):
    def __init__(self,config,text_encoder):
        super().__init__()
        self.config = config
        self.text_encoder = BertModel.from_pretrained(text_encoder, add_pooling_layer=False)

        self.textual_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, config['category'])
        )

    def forward(self, text, targets=None, train=True):

        textual_resp = self.text_encoder(text.input_ids,
                                            attention_mask=text.attention_mask,
                                            return_dict=True)
        te_hiddens = textual_resp.last_hidden_state[:, 0, :]
        prediction = self.textual_cls_head(te_hiddens)

        if train:
            loss = F.cross_entropy(prediction, targets, ignore_index=-1)
            return loss
        else:
            return prediction
