import time
import torch
import torch.nn as nn
from models.backbone import get_backbone
from losses import cgi_loss, cpa_loss, loss_scheduler, focal_loss
import numpy as np
import logging
import os
from utils import relationship_learning
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class TransferNet(nn.Module):
    def __init__(self, args):
        super(TransferNet, self).__init__()
        # define the network
        # get the feature extractor and the pretrained head
        self.args = args
        self.num_class = args.num_class
        self.base_network = get_backbone(args.model_name).cuda()

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, self.num_class, bias=False))
        self.classifier_layer.apply(weights_init_classifier)

        # define the loss functions
        self.cpa_loss = cpa_loss.CPALoss(self.num_class)
        self.cgi_loss = cgi_loss.CGILoss()
        if args.clf_loss == "cross_entropy":
            self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        elif args.clf_loss == "focal_loss":
            self.clf_loss = focal_loss.FocalLoss(num_classes=self.num_class,smoothing=args.label_smoothing)

        # prepare the embedding of the source domain on the imagenet pretrained task
        self.embedding = self.get_embedding(args)

        # prepare the dynamic coefficient for loss functions
        self.lamb = loss_scheduler.LambdaSheduler(max_iter=args.max_iter)

    def get_embedding(self,args):
        embedding_name = args.datasets + "_" + args.src_domain + "_" + args.model_name + ".npy"
        root = os.path.join(args.log_dir,"embedding")
        if not os.path.exists(root):
            os.makedirs(root)
        embedding_path = os.path.join(root,embedding_name)
        if not os.path.exists(embedding_path):
            print("The first time training for the src_domain. It need few minutes to learn the embedding~")
            print("Embedding learning...")
            data_path = os.path.join(args.data_dir, args.src_domain)
            embedding = relationship_learning.get_embedding(args,data_path, self.base_network, embedding_path)
        else:
            print("Loaded embedding...")
            embedding = torch.tensor(np.load(embedding_path)).cuda()
        print("Embedding acquirement finished! ")
        print(f"Embedding shape: {embedding.size()}")
        time.sleep(1)
        return embedding

    def forward(self, source, target, source_label, target_strong=None, label_set=None):
        source = self.base_network.forward_features(source)

        # calculate source classification loss Lclf
        source_logits = self.classifier_layer(self.base_network.flatten(source))
        clf_loss = self.args.clf_factor * self.clf_loss(source_logits, source_label)

        if not self.args.baseline:
            target = self.base_network.forward_features(target)
            # calculate calibrated probability alignment loss Lcpa
            target_imagenet_logits = self.base_network.forward_head(target)
            target_logits = self.classifier_layer(self.base_network.flatten(target).detach())
            # target_logits = self.classifier_layer(self.base_network.flatten(target))
            source_imagenet_logits = self.base_network.forward_head(source)
            if self.args.cpa_factor >0.0:
                cpa_loss = self.cpa_loss(source_imagenet_logits, target_imagenet_logits, source_label, target_logits, self.embedding, label_set)
            else:
                cpa_loss = 0.0

            if self.args.cgi_factor>0.0:
            # calculate calibrated gini impurity loss Lcgi
                cgi_loss = self.cgi_loss(target_logits, target_imagenet_logits, self.embedding, label_set)
            else:
                cgi_loss = 0.0

            # calculate the lamb and update
            lamb = self.lamb.lamb()
            self.lamb.step()
            transfer_loss = lamb*(self.args.cpa_factor*cpa_loss + self.args.cgi_factor*cgi_loss)
        else:
            transfer_loss = torch.tensor(0).to(source_label.device)

        if self.args.fixmatch and target_strong is not None:
            max_prob, pred_u = torch.max(F.softmax(target_logits), dim=-1)
            target_strong = self.base_network.forward_features(target_strong)
            target_strong = self.classifier_layer(self.base_network.flatten(target_strong))
            fixmatch_loss = self.args.fixmatch_factor * (F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
                                                         max_prob.ge(self.args.threshold).float().detach()).mean()
            transfer_loss += fixmatch_loss

        return clf_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}
        ]
        return params

    def predict(self, x):
        features = self.base_network.forward_features(x)
        features = self.base_network.flatten(features)
        logit = self.classifier_layer(features)
        return logit