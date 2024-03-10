from re import A
from PIL import Image, ImageFile
import torch
import numpy as np
import os
import sys
import pickle
from numpy.testing import assert_array_almost_equal
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageFilter
from loguru import logger

###
class_num = 10
###
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Onehot(object):
    def __call__(self, sample, num_classes=10):
        target_onehot = torch.zeros(num_classes)
        target_onehot[sample] = 1

        return target_onehot

def train_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def train_aug_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

def query_transform():
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def load_data(source_list, target_list, batch_size, num_workers, task = 'cross'):

    usps.init(source_list, target_list, task)
    query_dataset = usps('query', query_transform(),target_transform=Onehot())
    train_s_dataset = usps('train_s', train_transform(),target_transform=Onehot())
    train_t_dataset = usps('train_t', train_transform(),target_transform=Onehot())
    retrieval_dataset = usps('retrieval', query_transform(), target_transform=Onehot())
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_s_dataloader = DataLoader(
        train_s_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_t_dataloader = DataLoader(
        train_t_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, train_s_dataloader, train_t_dataloader, retrieval_dataloader


class usps(Dataset):
    def __init__(self, mode,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if mode == 'train_s':
            self.data = usps.TRAIN_S_DATA
            self.targets = usps.TRAIN_S_TARGETS

        elif mode == 'train_t':
            self.data = usps.TRAIN_T_DATA
            self.targets = usps.TRAIN_T_TARGETS

        elif mode == 'query':
            self.data = usps.QUERY_DATA
            self.targets = usps.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = usps.RETRIEVAL_DATA
            self.targets = usps.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.target_transform(self.targets[index]), index

    def __len__(self):
        return self.data.shape[0]
    
    def get_targets(self):
        one_hot = torch.zeros((self.targets.shape[0],10))
        for i in range(self.targets.shape[0]):
            one_hot[i,:] = self.target_transform(self.targets[i])
        return  one_hot


    @staticmethod
    def init(source_list, target_list, task):
        source_data = []
        source_label = []
        target_data = []
        target_label = []
        
        with open(source_list, 'r') as f:
            for line in f.readlines():
                source_data.append(line.split()[0].replace(\
                    '/workspace/caozhangjie','/path_to_data'))
                source_label.append(int(line.split()[1]))
        
        with open(target_list, 'r') as f:
            for line in f.readlines():
                target_data.append(line.split()[0].replace(\
                    '/workspace/caozhangjie','/path_to_data'))
                target_label.append(int(line.split()[1]))

        source_data = np.array(source_data)
        source_label = np.array(source_label)
        target_data = np.array(target_data)
        target_label = np.array(target_label)


        if task == 'cross':
            
            perm_index = np.random.permutation(target_data.shape[0])
            query_index = perm_index[:int(0.1*target_data.shape[0])]
            train_t_index = perm_index[int(0.1*target_data.shape[0]):]
            
            usps.QUERY_DATA = target_data[query_index]
            usps.QUERY_TARGETS = target_label[query_index]

            usps.TRAIN_S_DATA = source_data
            usps.TRAIN_S_TARGETS = source_label

            usps.TRAIN_T_DATA = target_data[train_t_index]
            usps.TRAIN_T_TARGETS = target_label[train_t_index]

            usps.RETRIEVAL_DATA = source_data
            usps.RETRIEVAL_TARGETS = source_label

            logger.info('Query Num: {}'.format(usps.QUERY_DATA.shape[0]))
            logger.info('Retrieval Num: {}'.format(usps.RETRIEVAL_DATA.shape[0]))
            logger.info('Source Train Num: {}'.format(usps.TRAIN_S_DATA.shape[0]))
            logger.info('Target Train Num: {}'.format(usps.TRAIN_T_DATA.shape[0]))
            

