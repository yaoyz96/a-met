import torch
import torch.nn as nn

import math


backbones = {}

def register(name):
  def decorator(cls):
    backbones[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if name in backbones:
    backbone = backbones[name](**kwargs)
  else:
    raise ValueError("Unknown backbone {:s}".format(name))
  if torch.cuda.is_available():
    backbone.cuda()
  return backbone


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
