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

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# It takes a list of strings and returns a list of one-hot vectors
class Onehot(object):
    def __call__(self, sample, num_class=65):
        target_onehot = torch.zeros(num_class)
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
    """
    It loads the data from the source and target domains, and returns the data loaders for the query,
    train_s, train_t, and retrieval datasets
    
    :param source_list: the list of source domains
    :param target_list: the target domain
    :param batch_size: The number of images to be considered in each batch
    :param num_workers: the number of threads to use for loading the data
    :param task: the task to be performed, either 'cross' or 'self', defaults to cross (optional)
    """

    officehome.init(source_list, target_list, task)
    query_dataset = officehome('query', query_transform(),target_transform=Onehot())
    train_s_dataset = officehome('train_s', train_transform(),target_transform=Onehot())
    train_t_dataset = officehome('train_t', train_transform(),target_transform=Onehot())
    retrieval_dataset = officehome('retrieval', query_transform(), target_transform=Onehot())
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


# This class is used to load the officehome dataset.
class officehome(Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        """
        The function takes in the mode, transform, and target_transform as arguments. It then sets the
        transform and target_transform to the arguments passed in. It then sets the mode to the mode
        passed in. It then sets the number of classes to 65. It then checks the mode and sets the data
        and targets to the appropriate data and targets
        
        :param mode: train_s, train_t, query, retrieval
        :param transform: a function that takes in an PIL image and returns a transformed version
        :param target_transform: a function/transform that takes in the target and transforms it
        """
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.num_class = 65

        if mode == 'train_s':
            self.data = officehome.TRAIN_S_DATA
            self.targets = officehome.TRAIN_S_TARGETS

        elif mode == 'train_t':
            self.data = officehome.TRAIN_T_DATA
            self.targets = officehome.TRAIN_T_TARGETS

        elif mode == 'query':
            self.data = officehome.QUERY_DATA
            self.targets = officehome.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = officehome.RETRIEVAL_DATA
            self.targets = officehome.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        """
        It takes an image, transforms it, and returns the transformed image, the target, and the index
        of the image
        
        :param index: The index of the image in the dataset
        :return: The image, the target, and the index.
        """
        img = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.target_transform(self.targets[index]), index

    def __len__(self):
        """
        The function returns the number of rows in the dataframe
        :return: The number of rows in the data.
        """
        return self.data.shape[0]
    
    def get_targets(self):
        """
        It takes the targets, and transforms them into one-hot vectors
        """
        one_hot = torch.zeros((self.targets.shape[0],self.num_class))
        for i in range(self.targets.shape[0]):
            one_hot[i,:] = self.target_transform(self.targets[i])
        return  one_hot


    @staticmethod
    def init(source_list, target_list, task):
        """
        > The function takes in the source and target data lists, and the task type, and returns the
        data and labels for the source and target domains
        
        :param source_list: the path to the source data list
        :param target_list: the path to the target data list
        :param task: 'cross' or 'self'
        """
        source_data = []
        source_label = []
        target_data = []
        target_label = []
        
        with open(source_list, 'r') as f:
            for line in f.readlines():
                source_data.append(line.split()[0].replace(\
                    '/data','/data/luoxiao/hash_project/CDAN/data'))
                source_label.append(int(line.split()[1]))
        
        with open(target_list, 'r') as f:
            for line in f.readlines():
                target_data.append(line.split()[0].replace(\
                    '/data','/data/luoxiao/hash_project/CDAN/data'))
                target_label.append(int(line.split()[1]))

        source_data = np.array(source_data)
        source_label = np.array(source_label)
        target_data = np.array(target_data)
        target_label = np.array(target_label)

        if task == 'cross':
            
            perm_index = np.random.permutation(target_data.shape[0])
            query_index = perm_index[:int(0.1*target_data.shape[0])]
            train_t_index = perm_index[int(0.1*target_data.shape[0]):]
            
            officehome.QUERY_DATA = target_data[query_index]
            officehome.QUERY_TARGETS = target_label[query_index]

            officehome.TRAIN_S_DATA = source_data
            officehome.TRAIN_S_TARGETS = source_label

            officehome.TRAIN_T_DATA = target_data[train_t_index]
            officehome.TRAIN_T_TARGETS = target_label[train_t_index]

            officehome.RETRIEVAL_DATA = source_data
            officehome.RETRIEVAL_TARGETS = source_label
