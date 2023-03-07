import sys
sys.path.append("..") 

import math
import torch
import torch.nn as nn
import models
import backbones
import torch.nn.functional as F

from .process import *
from .config_networks import ConfigureNetwork
from .models import register

from typing import Tuple

import matplotlib.pyplot as plt

@register('cross-attention')
class CrossAttention(nn.Module):
  def __init__(self, backbone, backbone_args, meta_args, mtd_args, **kwargs):
    super(CrossAttention, self).__init__()

    network = ConfigureNetwork(model="meta-baseline", 
                              backbone=backbone, 
                              backbone_args=backbone_args, 
                              meta_args=meta_args,
                              mtd_args=mtd_args, 
                            )
    self.backbone = network.get_backbone()
    self.attention = network.get_mtd_opt()

    self.out_dim = self.backbone.out_dim
  
  def forward(self, support_images: torch.Tensor, query_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through the model for one episode.
    :param support_images: (torch.tensor) Images of the support set (M x C x H x W), where M = batch_size x n_way x n_shot.
    :param query_images: (torch.tensor) Images of the query set (N x C x H x W), where N = batch_size x n_way x n_query.
    :return: (torch.tensor) Extracted feature of support set (M x feature_dim) and query set (N x feature_dim).
    """
    self.n_way = 20
    self.n_shot = 1
    self.n_query = 15

    # extract features of the support and query
    support_features = self._get_features(support_images)  # (n_shot*n_way, feat_dim)
    query_features = self._get_features(query_images)      # (n_query*n_way, feat_dim)

    # create fake support label
    y_shot = torch.from_numpy(np.repeat(range(self.n_way), self.n_shot)).cuda()

    x = torch.cat((support_features, query_features), dim=0)
    tx = self.attention(x, self.n_shot+self.n_query, y_shot)

    support_features, query_features = tx[:self.n_way*self.n_shot], tx[self.n_way*self.n_shot:]

    return support_features, query_features
  
  def _get_features(self, images) -> torch.Tensor:
    """
    Helper function to extract feature representation for each image.
    :param images: (torch.tensor) Images in the set (N x C x H x W).
    :return: (torch.tensor) Feature representation for each images (N x feature_dim).
    """
    return self.backbone(images)   # (N x feature_dim)

  def forward(self, ):
    x = x.contiguous().view(-1, *x.size()[2:])  # [batch_size*n_way*(n_shot+n_query), C, H, W]
    x = self.backbone(x)  # [batch_size*n_way*(n_shot+n_query), feat_dim]

    # create fake query label
    y_shot = torch.from_numpy(np.repeat(range(self.n_way), self.n_shot)).cuda()
    y_shot = y_shot.repeat(self.batch_size)

    x = x.contiguous().view(self.batch_size, -1, x.size()[-1]).squeeze(0)  # drop batch size dimension
    tx = self.attention(x, self.n_shot+self.n_query, y_shot)

    scores, label = self.simi.process(tx)

    return scores, label