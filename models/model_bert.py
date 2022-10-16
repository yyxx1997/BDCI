from transformers import BertModel, logging as tl
from torch import nn
import torch.nn.functional as F
from models.tricks import AS_Softmax, compute_kl_loss, MultiCEFocalLoss

tl.set_verbosity_error()
class BaselineBert(nn.Module):
    def __init__(self,config,text_encoder):
        super().__init__()
        self.config = config
        self.text_encoder = BertModel.from_pretrained(text_encoder, add_pooling_layer=False)
        self.weight_class = config['weights']
        self.focal_loss = MultiCEFocalLoss(class_num=config['category'], alpha=self.weight_class)
        self.textual_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, config['category'])
        )

    def forward(self, batch):

        text = batch['text']
        targets = batch['label']

        textual_resp = self.text_encoder(text.input_ids,attention_mask=text.attention_mask, return_dict=True)
        te_hiddens = textual_resp.last_hidden_state[:, 0, :]
        prediction = self.textual_cls_head(te_hiddens)
        
        output = {}
        output['prediction'] = prediction
        if self.training:
            loss = self.focal_loss(prediction, targets)
            output['loss'] = loss
        return output
