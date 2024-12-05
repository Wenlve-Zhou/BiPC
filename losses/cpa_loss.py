import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CPALoss(nn.Module):
    def __init__(self, num_class):
        super(CPALoss, self).__init__()
        self.num_class = num_class
        self.b = [i for i in range(65)]

    def get_train_index(self,preds,label_set):
        index = []
        for pred in preds:
            if pred in label_set:
                index.append(1)
            else:
                index.append(0)
        return torch.tensor(index).cuda()

    def forward(self, source, target, source_label, target_logits, embedding, label_set):
        if label_set is not None:
            label_set = [x for x in label_set if x not in self.b]
        coe = self.calibrated_coefficient(source_label, target_logits, label_set)
        loss = self.probability_alignment(source,target)
        loss = torch.sum(coe * loss)
        reg = self.regularize_term(source, source_label, embedding)
        return loss + reg

    def get_probability_distance(self,source,target):
        source = F.softmax(source, dim=-1)
        target = F.softmax(target, dim=-1)
        mixture = 0.5 * (source + target)
        return -0.5 * (source * mixture.log() + target * mixture.log())

    def probability_alignment(self,source,target):
        b_s, c = source.size()
        b_t, c = target.size()
        source_expand = source.unsqueeze(1).expand(b_s, b_t, c)
        target_expand = target.unsqueeze(0).expand(b_s, b_t, c)
        distance = self.get_probability_distance(source_expand,target_expand)
        return distance.sum(-1)

    def calibrated_coefficient(self, source_label, target_logits, label_set):
        # label for source
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label]  # one hot
        if label_set is not None:
            source_label_onehot[:,label_set] = 0

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum  # label ratio

        # Pseudo label
        target_logits = F.softmax(target_logits,dim=-1)
        target_label = target_logits.cpu().data.max(1)[1].numpy()
        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        # Cal weight
        set_s = set(source_label)
        set_t = set(target_label)

        weight_st = np.dot(source_label_onehot, target_logits.T)
        source_label_onehot_ = np.eye(self.num_class)[source_label]
        target_label_onehot_ = np.eye(self.num_class)[target_label]
        index_st = np.dot(source_label_onehot_, target_label_onehot_.T)
        weight_st = weight_st * index_st

        set_st = list(set(set_s) & set(set_t))
        length_st = len(set_st)

        if length_st != 0:
            weight_st = weight_st / length_st

        weight_st = weight_st.astype('float32')
        return torch.from_numpy(weight_st).cuda()

    def regularize_term(self,source, source_label,embedding):
        def cal_weight(label):
            label = label.cpu().data.numpy()
            label_onehot = np.eye(self.num_class)[label]  # one hot
            label_sum = np.sum(label_onehot, axis=0).reshape(1, self.num_class)
            label_sum[label_sum == 0] = 100
            label_onehot = label_onehot / label_sum  # label ratio
            label_onehot = label_onehot.astype('float32')
            return torch.from_numpy(label_onehot).cuda()

        b = source.size(0)
        emb_expand = embedding.unsqueeze(0).repeat(b, 1, 1)
        source_expand = source.unsqueeze(1).repeat(1, self.num_class, 1)
        source_expand_logsoftmax = F.log_softmax(source_expand, dim=-1)
        partial_loss = (-emb_expand * source_expand_logsoftmax).sum(-1)
        weight = cal_weight(source_label)
        partial_loss = (weight * partial_loss).sum(-1)
        return partial_loss.mean()

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    b = [2, 4]

    result = [x for x in a if x not in b]
    print(result)

