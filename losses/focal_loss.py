import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 num_classes=12,
                 smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        self.num_classes = num_classes
        self.smoothing = smoothing

    def label_smoothing(self, logits, label):
        lb_pos, lb_neg = 1. - self.smoothing, self.smoothing / (self.num_classes - 1)
        lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        return lb_one_hot

    def forward(self, logits, label):
        label_onehot = F.one_hot(label,num_classes=self.num_classes)
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label_onehot - probs).pow(self.gamma).neg()
        if self.smoothing>0.0:
            label = self.label_smoothing(logits, label)
        else:
            label = label_onehot
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
