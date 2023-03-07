import sys
sys.path.append("..") 

import torch
import torch.nn as nn

from .process import *
from .config_networks import ConfigureNetwork
from .models import register

from typing import Tuple

@register('baseline')
class Baseline(nn.Module):
  """
  A plain baseline architecture, 
  contains a backbone and a classifier (fc layer).
  Data flow:
    input images (BS x C x H x W), where BS = batch_size
    -> backbone(i.e. encoder), outputs feature (BS x feature_dim)
    -> classifier(i.e. FC layer), outputs logits (BS x num_classes)
  """
  def __init__(self, backbone: str, backbone_args: dict, classifier: str, classifier_args: dict, **kwargs):
    super(Baseline, self).__init__()

    network = ConfigureNetwork(model="baseline", 
                              backbone=backbone, 
                              backbone_args=backbone_args, 
                              classifier=classifier,
                              classifier_args=classifier_args,
                            )
    self.backbone = network.get_backbone()
    self.classifier = network.get_classifier()
  
  def forward(self, images: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the model for one iteration.
    :param images: (torch.tensor) Input images (BS x C x H x W).
    :return: (torch.tensor) Predicted logits (BS x num_classes).
    """
    features = self._get_features(images)
    logits = self._classify(features)
    
    return logits
  
  def _get_features(self, images: torch.Tensor) -> torch.Tensor:
    """
    Helper function to extract feature representation for each image.
    :param images: (torch.tensor) Images with one batch size (BS x C x H x W).
    :return: (torch.tensor) Feature representation for each images (BS x feature_dim).
    """
    return self.backbone(images).contiguous()

  def _classify(self, features: torch.Tensor) -> torch.Tensor:
    """
    Helper function to get classification scores for each image.
    :param features: (torch.tensor) Feature representation for each images (BS x feature_dim).
    :return: (torch.tensor) Predicted logits (BS x num_classes).
    """
    return self.classifier(features).contiguous()


@register('meta-baseline')
class MetaBaseline(nn.Module):
  """
  A plain meta-baseline architecture, 
  only contains a feature extractor (backbone).
  Data flow:
    input images (BS x C x H x W), BS = batch_size
    -> backbone(i.e. encoder), outputs feature (BS x feature_dim)
  """
  def __init__(self, backbone: str, backbone_args: dict, meta_args: dict, **kwargs):
    super(MetaBaseline, self).__init__()

    network = ConfigureNetwork(model="meta-baseline", 
                              backbone=backbone, 
                              backbone_args=backbone_args, 
                              meta_args=meta_args, 
                            )
    self.backbone = network.get_backbone()
    self.out_dim = self.backbone.out_dim
    
  def forward(self, support_images: torch.Tensor, query_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through the model for one episode.
    :param support_images: (torch.tensor) Images of the support set (M x C x H x W), where M = batch_size x n_way x n_shot.
    :param query_images: (torch.tensor) Images of the query set (N x C x H x W), where N = batch_size x n_way x n_query.
    :return: (torch.tensor) Extracted feature of support set (M x feature_dim) and query set (N x feature_dim).
    """
    # extract features of the support and query
    support_features = self._get_features(support_images)
    query_features = self._get_features(query_images)

    return support_features, query_features
  
  def _get_features(self, images) -> torch.Tensor:
    """
    Helper function to extract feature representation for each image.
    :param images: (torch.tensor) Images in the set (N x C x H x W).
    :return: (torch.tensor) Feature representation for each images (N x feature_dim).
    """
    return self.backbone(images)   # (N x feature_dim)

