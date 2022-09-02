from functools import partial
from transformers import BertModel
import torch
from torch import nn
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class BaselineBert(nn.Module):
    def __init__(self,config,text_encoder):
        super().__init__()
        self.config = config
        self.text_encoder = BertModel.from_pretrained(text_encoder, add_pooling_layer=False)
        self.r_drop_rate = self.config['r_drop']
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

            if self.r_drop_rate:
                textual_resp_2 = self.text_encoder(text.input_ids,
                                            attention_mask=text.attention_mask,
                                            return_dict=True)
                te_hiddens_2 = textual_resp_2.last_hidden_state[:, 0, :]
                prediction_2 = self.textual_cls_head(te_hiddens_2)

                # cross entropy loss for classifier
                loss_2 = F.cross_entropy(prediction_2, targets, ignore_index=-1)
                ce_loss = 0.5 * (loss + loss_2)
                kl_loss = compute_kl_loss(prediction, prediction_2)

                # carefully choose hyper-parameters
                loss = ce_loss + self.r_drop_rate * kl_loss

            return loss
        else:
            return prediction
