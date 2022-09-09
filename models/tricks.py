import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AS_Softmax(nn.Module):
    """
    AS-Softmax:
        Training objective:pt − pi ≥ δ;
        δ ∈ [0, 1], if δ=1, AS-Softmax is equivalent to softmax.
    """

    def __init__(self, delta=0.10):

        super(AS_Softmax, self).__init__()
        self.delta = delta

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor):  Logits refers to the output of the model after passing through the classifier. Its size is (batch, labels_num).
            labels (torch.Tensor):  Labels refers to the gold labels of input, and its size is (batch).
        Returns:
            as_logits (torch.Tensor):  Logits after AS-Softmax algorithm processing
            labels (torch.Tensor)
        """
        if not self.delta:
            return logits, labels
            
        active_logits = logits.view(-1, logits.size(-1))
        active_labels = labels.view(-1)
        logits_softmax = nn.Softmax(dim=-1)(active_logits)
        gold_softmax = torch.gather(logits_softmax, dim=1, index=labels.unsqueeze(-1))

        is_lt = (gold_softmax.repeat(1, active_logits.shape[1]) - logits_softmax) <= self.delta
        as_logits = torch.where(is_lt, active_logits, torch.tensor(float('-inf')).type_as(active_logits))
        return as_logits, active_labels

def assoftmax_loss(logits, targets, theta=0.2):
    logits_softmax = nn.Softmax(dim=-1)(logits)
    target_prob = torch.gather(logits_softmax, dim=1, index=targets.unsqueeze(-1))
    zi = (target_prob.repeat(1, logits.size(-1)) - logits_softmax) <= theta
    as_z = torch.where(zi, logits, torch.tensor(float('-inf')).type_as(logits))
    loss = F.cross_entropy(as_z, targets, ignore_index=-1)
    return loss

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


class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)].to(predict.device) # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss