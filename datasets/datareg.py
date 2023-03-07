import os
import pandas as pd

from torchvision.datasets import ImageFolder

from .datasets import register


@register('mini_imagenet')
class MiniImageNet:
  """
  data structure of 'mini-imagenet':
    -| miniImageNet
    ---| images
    -----| n0153282900000005.jpg
    -----| n0153282900000006.jpg
    -----| n0153282900000007.jpg
    -----| ...
    ---| labels
    -----| fold0_train.csv
    -----| fold0_val.csv
    -----| ...
    ---| train.csv
    ---| val.csv
    ---| test.csv
  """
  def __init__(self, data_path, split='train', fold=-1):
    super(MiniImageNet, self).__init__()
    self.data = []
    self.label = []
    if fold != -1:
      data_split = os.path.join(data_path, 'labels')
      data_split += '/fold' + str(fold) + '_' + split + '.csv'
    else: 
      data_split = data_path + '/' + split + '.csv'
    data_list = pd.read_csv(data_split)
    data_list = data_list.values.tolist()

    for d in data_list:
      img = os.path.join(data_path, 'images', d[0])  # image path
      self.data.append(img)
      self.label.append(d[1])
    self.label_idx = convertLabel(self.label)  # convert string label to index
  
  def getLabel(self):
    self.label_str = idx2label(self.label)
    return self.label_str

  def getData(self):
    return self.data, self.label_idx   # return data/label path list


@register('tiered_imagenet')
class TieredImageNet:
  """
  data structure of 'tiered-imagenet':
    -| tieredImageNet
    ---| train
    -----| n01530575
    -------| n0153057500000001.jpg
    -------| ...
    -----| n01531178
    -------| ...
    ---| val
    ---| test
    ---| train.csv
    ---| val.csv
    ---| test.csv
  """
  def __init__(self, data_path, split='train', fold=-1):
    super(TieredImageNet, self).__init__()
    self.data = []
    self.label = []
    
    data_split = data_path + '/' + split + '.csv'
    data_list = pd.read_csv(data_split)
    data_list = data_list.values.tolist()

    for d in data_list:
      img = os.path.join(data_path, split, d[1], d[0])  # image path
      self.data.append(img)
      self.label.append(d[1])
    self.label_idx = convertLabel(self.label)  # convert string label to index
  
  def getLabel(self):
    self.label_str = idx2label(self.label)
    return self.label_str

  def getData(self):
    return self.data, self.label_idx   # return data/label path list


@register('omniglot')
class Omniglot:
  """
  data structure of 'omniglot':
    -| Omniglot
    ---| Alphabet_of_the_Magi (super-class 1)
    -----| character01
    -------| 0709_01.png
    -------| 0709_02.png
    -------| ...
    -----| character02
    -----| ...
    ---| Angelic (super-class 2)
    -------| ...
  """
  def __init__(self, data_path, split='train', fold=-1):
    super(Omniglot, self).__init__()
    self.data = []
    self.label = []

    data_cls_split = data_path + '/' + split + '.csv'
    data_cls_list = pd.read_csv(data_cls_split)
    data_cls_list = data_cls_list.values.tolist()

    for d in data_cls_list:
      img_path = os.path.join(data_path, 'data', d[0])
      self.data.append(img_path)
      self.label.append(d[1])
    
    self.label_idx = convertLabel(self.label)

  def getLabel(self):
    self.label_str = idx2label(self.label)
    return self.label_str
  
  def getData(self):
    return self.data, self.label_idx


@register('cub')
class CUB2011:
  """
  CUB-200-2011, fine-grained classification.
  """
  def __init__(self, data_path, split='train'):
    super(CUB2011, self).__init__()
    self.data = []
    self.label = []

    data_path = data_path + '/images'
    label_list = os.listdir(data_path)
    for label in label_list:
      images_path = os.path.join(data_path, label)
      images_list = os.listdir(images_path)
      for img in images_list:
        img_path = os.path.join(images_path, img)
        self.data.append(img_path)
        self.label.append(label)

    self.label_idx = convertLabel(self.label)

  def getLabel(self):
    self.label_str = idx2label(self.label)
    return self.label_str
  
  def getData(self):
    return self.data, self.label_idx


def idx2label(label):
    label_str = {}
    num = len(label)
    k = 0
    for i in range(num):
        if i == 0:
            label_str[i] = label[i]
        else:
            if label[i] != label[i-1]:
                k += 1
                label_str[k] = label[i]
    return label_str


def convertLabel(label):
    norm_label = []
    num = len(label)
    k = 0
    for i in range(num):
        if i == 0:
            norm_label.append(k)
        else:
            if label[i] != label[i-1]:
                k += 1
            norm_label.append(k)
    return norm_label


if __name__ == '__main__':
  pass
