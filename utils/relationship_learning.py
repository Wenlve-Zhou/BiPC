import torch
import torch.nn as nn
import numpy as np

from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import ssl
import warnings
from utils.data_loader import toRGB

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

__all__ = ['relationship_learning', 'direct_relationship_learning']

import torch.multiprocessing
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

def calibrate(logits, labels):
    """
    calibrate by minimizing negative log likelihood.
    :param logits: pytorch tensor with shape of [N, N_c]
    :param labels: pytorch tensor of labels
    :return: float
    """
    scale = nn.Parameter(torch.ones(
        1, 1, dtype=torch.float32), requires_grad=True)
    optim = torch.optim.LBFGS([scale])

    def loss():
        optim.zero_grad()
        lo = nn.CrossEntropyLoss()(logits * scale, labels)
        lo.backward()
        return lo

    state = optim.state[scale]
    for i in range(20):
        optim.step(loss)
        if state['n_iter'] < optim.state_dict()['param_groups'][0]['max_iter']:
            break

    return scale.item()


def softmax_np(x):
    max_el = np.max(x, axis=1, keepdims=True)
    x = x - max_el
    x = np.exp(x)
    s = np.sum(x, axis=1, keepdims=True)
    return x / s


def relationship_learning(train_logits, train_labels, validation_logits, validation_labels):
    """
    :param train_logits (ImageNet logits): [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param train_labels:  [N], where 0 <= each number < N_t, and N_t is the number of target dataset
    :param validation_logits (ImageNet logits): [N, N_p]
    :param validation_labels:  [N]
    :return: [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """

    # convert logits to probabilities
    train_probabilities = softmax_np(train_logits * 0.8840456604957581)
    validation_probabilities = softmax_np(
        validation_logits * 0.8840456604957581)

    all_probabilities = np.concatenate(
        (train_probabilities, validation_probabilities))
    all_labels = np.concatenate((train_labels, validation_labels))

    Cs = []
    accs = []
    classifiers = []
    for C in tqdm(iterable=[1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4], desc="Relationship Learning"):
        cls = LogisticRegression(
            multi_class='multinomial', C=C, fit_intercept=False)
        cls.fit(train_probabilities, train_labels)
        val_predict = cls.predict(validation_probabilities)
        val_acc = np.sum((val_predict == validation_labels).astype(
            np.float64)) / len(validation_labels)
        Cs.append(C)
        accs.append(val_acc)
        classifiers.append(cls)

    accs = np.asarray(accs)
    ind = int(np.argmax(accs))
    cls = classifiers[ind]
    del classifiers

    validation_logits = np.matmul(validation_probabilities, cls.coef_.T)
    validation_logits = torch.from_numpy(validation_logits.astype(np.float64))
    validation_labels = torch.from_numpy(validation_labels)

    scale = calibrate(validation_logits, validation_labels)

    p_target_given_pretrain = softmax_np(
        cls.coef_.T * scale)  # shape of [N_p, N_c], conditional probability p(target_class | pre-trained class)

    # in the paper, both ys marginal and yt marginal are computed
    # here we only use ys marginal to make sure p_pretrain_given_target is a valid conditional probability
    # (make sure p_pretrain_given_target[i] sums up to 1)
    pretrain_marginal = np.mean(all_probabilities, axis=0).reshape(
        (-1, 1))  # shape of [N_p, 1]
    p_joint_distribution = (p_target_given_pretrain * pretrain_marginal).T
    p_pretrain_given_target = p_joint_distribution / \
        np.sum(p_joint_distribution, axis=1, keepdims=True)

    return p_pretrain_given_target


def direct_relationship_learning(all_logits, all_labels):
    """
    The direct approach of learning category relationship.
    :param train_logits (ImageNet logits): [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param train_labels:  [N], where 0 <= each number < N_t, and N_t is the number of target dataset
    :param validation_logits (ImageNet logits): [N, N_p]
    :param validation_labels:  [N]
    :return: [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """
    # convert logits to probabilities
    all_probabilities = softmax_np(all_logits * 0.8840456604957581)

    N_t = np.max(all_labels) + 1 # the number of target classes
    conditional = []
    for i in tqdm(iterable=range(N_t),desc="Relationship Learning"):
        this_class = all_probabilities[all_labels == i]
        average = np.mean(this_class, axis=0, keepdims=True)
        conditional.append(average)
    return np.concatenate(conditional)


def get_feature(net,loader,dec="train"):
    train_labels_list = []
    imagenet_labels_list = []

    for train_inputs, train_labels in tqdm(iterable=loader, desc=f"Feature Extraction for {dec}"):
        net.eval()
        train_labels_list.append(train_labels)

        train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
        imagenet_labels = net(train_inputs)
        imagenet_labels = imagenet_labels.detach().cpu().numpy()

        imagenet_labels_list.append(imagenet_labels)

    all_train_labels = np.concatenate(train_labels_list, 0)
    all_imagenet_labels = np.concatenate(imagenet_labels_list, 0)

    return all_imagenet_labels, all_train_labels


class Folder(datasets.ImageFolder):
    def __init__(self,root,transform=None,train=True):
        super(Folder, self).__init__(root=root,transform=transform)
        train_imgs,valid_imgs = self.split(self.samples)
        self.samples = train_imgs if train else valid_imgs

    def split(self,all,ratio=0.5):
        all = np.array(all)
        train,valid = [], []
        for c in range(len(self.classes)):
            idx = np.where(all[:,1]==str(c))[0]
            np.random.shuffle(idx)
            train_num = int(len(idx)*ratio)
            train_idx,valid_idx = idx[:train_num], idx[train_num:]
            train.extend(all[train_idx].tolist())
            valid.extend(all[valid_idx].tolist())
        train = self.str2int(train)
        valid = self.str2int(valid)
        return train,valid

    def str2int(self,lists):
        for i in range(len(lists)):
            lists[i] = [lists[i][0],int(lists[i][1])]
        return lists

def digit_split(data,label,classes_num=10,ratio=0.5):
    data = np.array(data)
    label = np.array(label)
    train_data,train_label,valid_data, valid_label = [], [], [], []
    for c in range(classes_num):
        idx = np.where(label==c)[0]
        np.random.shuffle(idx)
        train_num = int(len(idx)*ratio)
        train_idx,valid_idx = idx[:train_num], idx[train_num:]
        train_data.extend(data[train_idx].tolist())
        train_label.extend(label[train_idx].tolist())
        valid_data.extend(data[valid_idx].tolist())
        valid_label.extend(label[valid_idx].tolist())
    return torch.tensor(train_data),torch.tensor(train_label), torch.tensor(valid_data), torch.tensor(valid_label)

class MNIST(datasets.MNIST):
    def __init__(self,root,transform=None,train=True):
        super(MNIST, self).__init__(root=root,transform=transform,train=True,download=True)
        train_data,train_label, valid_data, valid_label = digit_split(self.data, self.targets)
        if train:
            self.data, self.targets = train_data, train_label
        else:
            self.data, self.targets = valid_data, valid_label


class USPS(datasets.USPS):
    def __init__(self, root, transform=None, train=True):
        super(USPS, self).__init__(root=root, transform=transform,train=True,download=True)
        train_data, train_label, valid_data, valid_label = digit_split(self.data, self.targets)
        if train:
            self.data, self.targets = train_data, train_label
        else:
            self.data, self.targets = valid_data, valid_label
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)


class SVHN(datasets.SVHN):
    def __init__(self, root, transform=None, train=True):
        super(SVHN, self).__init__(root=root, transform=transform,split="train",download=True)
        train_data, train_label, valid_data, valid_label = digit_split(self.data, self.labels)
        if train:
            self.data, self.labels = train_data, train_label
        else:
            self.data, self.labels = valid_data, valid_label
        self.data = np.uint8(np.array(self.data))
        self.labels = np.array(self.labels)

def get_embedding(args,path, model,saved_path):
    if args.datasets == "visda":
        transform = transforms.Compose(
                [
                    transforms.Resize([256, 256]),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    elif args.datasets == "digits":
        transform = transforms.Compose(
            [
                toRGB(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])
    else:
        transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    if args.datasets == "digits":
        if "mnist" in path:
            train_dataset = MNIST(root=path,transform=transform,train=True)
            val_dataset = MNIST(root=path, transform=transform, train=False)
        elif "usps" in path:
            train_dataset = USPS(root=path,transform=transform,train=True)
            val_dataset = USPS(root=path, transform=transform, train=False)
        elif "svhn" in path:
            train_dataset = SVHN(root=path,transform=transform,train=True)
            val_dataset = SVHN(root=path, transform=transform, train=False)
    else:
        train_dataset = Folder(path, transform, train=True)
        val_dataset = Folder(path, transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False)
    train_imagenet_labels, train_train_labels = get_feature(model,train_loader,"train folder")
    val_imagenet_labels, val_train_labels = get_feature(model,val_loader, "valid folder")
    relationship = relationship_learning(train_imagenet_labels, train_train_labels,
                                         val_imagenet_labels, val_train_labels)
    np.save(saved_path,relationship)
    return torch.tensor(relationship).cuda()