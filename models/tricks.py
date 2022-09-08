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

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss