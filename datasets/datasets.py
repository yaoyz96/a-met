import os
import cv2
import sys
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
import utils.helpers as helpers

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .transform import TransformLoader
from .samplers import EpisodicSampler, EpisodicBatchSampler

from PIL import Image
from pathlib import Path

DEFAULT_ROOT = './materials'

datasets = {}

def register(name):
  def decorator(cls):
    datasets[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if kwargs.get('data_path') is None:
    kwargs['data_path'] = os.path.join(DEFAULT_ROOT, name)
  dataset = datasets[name](**kwargs)
  data, label = dataset.getData()
  label_name = dataset.getLabel()
  return data, label, label_name


class SimpleDataset(Dataset):
  """
  Read the whole dataset, i.e. whole dataset training manner.
  """
  def __init__(self, dataset, data_path, transform, split='train', fold=-1):
    super(SimpleDataset, self).__init__()
    data_params = dict(data_path=data_path, split=split, fold=fold)
    self.data, self.label, self.label_name = make(dataset, **data_params)
    self.n_classes = self.label[-1] + 1
    self.transform = transform
      
  def __getitem__(self, i):
    img = Image.open(self.data[i]).convert('RGB')
    return self.transform(img), self.label[i]

  def __len__(self):
    return len(self.data)


class MetaDataset(Dataset):
  """
  Return the dataset by classes index
  """
  def __init__(self, dataset, data_path, group, transform, split='train'):
    super(MetaDataset, self).__init__()

    data_params = dict(data_path=data_path, split=split)
    self.data, self.label, self.label_name = make(dataset, **data_params)
    self.n_classes = self.label[-1] + 1
    self.cls_list = np.unique(self.label).tolist()  # class list

    self.cls_pack = {}    # package of class
    for c in self.cls_list:
      self.cls_pack[c] = []
    for x, y in zip(self.data, self.label):
      self.cls_pack[y].append(x)  # group the dataset by class idx
    
    self.pack_loader = []
    pack_loader_params = dict(batch_size=group, shuffle=True, num_workers=0, pin_memory=False)
    for c in self.cls_list:
      pack_dataset = PackDataset(c, self.cls_pack[c], transform=transform)
      self.pack_loader.append(DataLoader(pack_dataset, **pack_loader_params))

  def __getitem__(self, i):
    return next(iter(self.pack_loader[i]))

  def __len__(self):
    return len(self.cls_list)   # number of classes


class PackDataset(Dataset):
  def __init__(self, c, cls_pack, transform=transforms.ToTensor()):
    self.cls_pack = cls_pack
    self.label = c
    self.transform = transform

  def __getitem__(self, i):
    img = Image.open(os.path.join(self.cls_pack[i])).convert('RGB')
    return self.transform(img), self.label

  def __len__(self):
    return len(self.cls_pack)  # number of images for one class


class SimpleDataLoader:
  """
  Data mamager for whole set training manner, i.e. the traditonal training process.
  """
  def __init__(self, dataset, data_path, batch_size, input_size, aug=False, split='train', fold=-1, device=None):
    super(SimpleDataLoader, self).__init__()

    self.dataset = dataset
    self.data_path = data_path

    self.batch_size = batch_size
    self.split = split
    
    self.aug = aug
    self.transform_loader = TransformLoader(input_size)
    transform = self.transform_loader.getTransform(aug)

    dataset = SimpleDataset(self.dataset, self.data_path, transform, self.split)
    helpers.print_and_log('\033[0;32mINFO\033[0m |==> {} dataset: {}(x{}), {} classes'
                          .format(self.split, self.dataset, len(dataset), dataset.n_classes))

    dl_params = dict(batch_size=self.batch_size, num_workers=8, pin_memory=True, prefetch_factor=500)
    dl_params['shuffle'] = True if aug else False
    self.data_loader = DataLoader(dataset, drop_last=True, **dl_params)

  def __iter__(self):
    return self._iterate_dataset()

  def __len__(self):
    return len(self.data_loader)

  def _iterate_dataset(self):
    data_loader = self.data_loader
    for _, (images, labels) in enumerate(data_loader):
      yield images, labels
  

class MetaDataLoader:
  """
  DataLoader for episode training manner, i.e. meta-learning
  """
  def __init__(self, dataset, data_path, n_way, n_shot, n_query, n_episode, input_size, batch_size=1, aug=False, split='train', fold=-1, device=None):
    super(MetaDataLoader, self).__init__()

    self.dataset = dataset
    self.data_path = data_path
    
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.n_episode = n_episode
    self.group = n_shot + n_query

    self.aug = aug
    self.batch_size = batch_size
    self.split = split
    self.device = device

    self.transform_loader = TransformLoader(input_size)
    transform = self.transform_loader.getTransform(aug)

    dataset = MetaDataset(self.dataset, self.data_path, self.group, transform, self.split)
    helpers.print_and_log('\033[0;32mINFO\033[0m |==> {} dataset: {}(x{}), {} classes'
                          .format(self.split, self.dataset, len(dataset.data), dataset.n_classes))

    if self.batch_size > 1:
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode, self.batch_size)
    else:
      sampler = EpisodicSampler(len(dataset), self.n_way, self.n_episode)
  
    # dataloader of the warped data
    dl_params = dict(num_workers=8, pin_memory=True)
    self.data_loader = DataLoader(dataset, batch_sampler=sampler, **dl_params)
  
  def __iter__(self):
    return self._iterate_dataset()

  def __len__(self):
    return self.n_episode
  
  def _iterate_dataset(self):
    data_loader = self.data_loader
    for _, (images, labels) in enumerate(data_loader):
      task_dict = {"support_images": images[:, :self.n_shot],
                  "support_labels": labels[:, :self.n_shot],
                  "query_images": images[:, self.n_shot:],
                  "query_labels": labels[:, self.n_shot:]
                }
      yield self._prepare_data(task_dict)
    
  def _prepare_data(self, task_dict):
    """
    support_images: (BS x n_way, n_shot, C, H, W)
    support_labels: (BS x n_way, n_shot)
    query_images: (BS x n_way, n_query, C, H, W)
    query_labels: (BS x n_way, n_query)
    """
    support_images, support_labels_fake = task_dict['support_images'], task_dict['support_labels']
    query_images, query_labels_fake = task_dict['query_images'], task_dict['query_labels']
    # create fake label, range form 0 to num_classes-1
    for idx in range(self.n_way):
      support_labels_fake[idx, :] = idx
      query_labels_fake[idx, :] = idx

    c, h, w = support_images.shape[-3], support_images.shape[-2], support_images.shape[-1]

    support_images = task_dict["support_images"].contiguous().view(-1, c, h, w)
    support_labels_fake = support_labels_fake.contiguous().view(-1)
    query_images = task_dict["query_images"].contiguous().view(-1, c, h, w)
    query_labels_fake = query_labels_fake.contiguous().view(-1)

    support_images = support_images.to(self.device)
    query_images = query_images.to(self.device)
    support_labels_fake = support_labels_fake.to(self.device)
    query_labels_fake = query_labels_fake.to(self.device)

    return support_images, query_images, support_labels_fake, query_labels_fake


class SimpleLargeDataLoader:
  """
  Dataloader of large datasets for whole set training manner, i.e. the traditonal training process.
  """
  def __init__(self, dataset, data_path, batch_size, input_size, aug=False, split='train', fold=-1, device_num=1):
    super(SimpleDataLoader, self).__init__()

    self.dataset = dataset
    self.data_path = data_path

    self.batch_size = batch_size
    self.split = split
    self.aug = aug

    # [TODO] config
    
  def __iter__(self):
    return self._iterate_dataset()

  def __len__(self):
    return len(self.data_loader)

  def _iterate_dataset(self):
    data_loader = self.data_loader
    for _, (images, labels) in enumerate(data_loader):
      yield images, labels