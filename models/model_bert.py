from transformers import BertModel
from torch import nn
import torch.nn.functional as F
from models.tricks import AS_Softmax, compute_kl_loss, MultiCEFocalLoss

class BaselineBert(nn.Module):
    def __init__(self,config,text_encoder):
        super().__init__()
        self.config = config
        self.text_encoder = BertModel.from_pretrained(text_encoder, add_pooling_layer=False)
        self.focal_loss = MultiCEFocalLoss(class_num=config['category'])
        self.textual_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, config['category'])
        )

    def forward(self, text, targets=None, train=True):

        output = {}
        textual_resp = self.text_encoder(text.input_ids,attention_mask=text.attention_mask,return_dict=True)
        te_hiddens = textual_resp.last_hidden_state[:, 0, :]
        prediction = self.textual_cls_head(te_hiddens)
        output['prediction'] = prediction
        if train:
            loss = self.focal_loss(prediction, targets)
            output['loss'] = loss
        return output
