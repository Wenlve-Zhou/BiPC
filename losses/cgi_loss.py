import torch.nn as nn
import torch
import torch.nn.functional as F

class CGILoss(nn.Module):

    def calibrated_coefficient(self, pred, pred_pretrained):
        distdance = F.kl_div(pred.log(), pred_pretrained, reduction='none').sum(-1)
        coe = torch.exp(-distdance).detach()
        return coe

    def calibrated_coefficient1(self, pred):
        epsilon = 1e-5
        H = -pred * torch.log(pred + epsilon)
        H = H.sum(dim=1)
        coe = torch.exp(-H).detach()
        return coe

    def gini_impurity(self,coe,pred):
        sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
        return torch.sum(coe * (1 - torch.sum(pred ** 2 / (sum_dim), dim=-1)))

    def get_pred_from_pretrained(self, logits, embedding):
        with torch.no_grad():
            logits = F.softmax(logits,dim=-1)
            dim, bs = embedding.size(0), logits.size(0)
            logits = logits.unsqueeze(1).repeat(1, dim, 1)
            embedding = embedding.unsqueeze(0).repeat(bs, 1, 1)
            distance = F.kl_div(logits.log(), embedding, reduction='none').sum(-1)
            distance = torch.exp(-distance)
            distance_sum = distance.sum(-1).unsqueeze(-1).repeat(1, dim)
            return (distance / distance_sum).detach()

    def forward(self, logits, target_imagenet_logits, embedding, label_set=None):
        pred_pretrained = self.get_pred_from_pretrained(target_imagenet_logits, embedding)
        pred = F.softmax(logits, dim=1)
        coe = self.calibrated_coefficient(pred, pred_pretrained)
        pred_mix = 0.5 * (pred+pred_pretrained)
        if label_set is not None:
            loss = self.gini_impurity(coe, pred[:,label_set]) + self.gini_impurity(1 - coe, pred_mix[:,label_set])
        else:
            loss = self.gini_impurity(coe, pred) + self.gini_impurity(1 - coe, pred_mix)
        return loss


