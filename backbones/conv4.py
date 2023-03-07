import torch
import torch.nn as nn

from .backbones import register


class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)


class Conv4(nn.Module):

  def __init__(self, in_dim, hid_dim, out_dim):
    super().__init__()

    self.in_dim = in_dim
    self.hid_dim = hid_dim
    self.out_dim = out_dim

    self.encoder = nn.Sequential(
      self._conv_block(in_dim, hid_dim),
      self._conv_block(hid_dim, hid_dim),
      self._conv_block(hid_dim, hid_dim),
      self._conv_block(hid_dim, out_dim),
      #Flatten()
    )

  def _conv_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
      )
    
  def forward(self, x):
    
    x = self.encoder(x)
    x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
    
    return x


@register('Conv4')
def conv4():
  return Conv4(3, 64, 64)