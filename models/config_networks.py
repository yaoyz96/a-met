import sys
sys.path.append("..")

import torch
import torch.nn as nn
import models
import backbones

from torch import dropout

from .sot import Attention

st1_model = ['baseline']
st2_model = ['meta-baseline']


class ConfigureNetwork:
  """ Creates the components of model, 
      including encoder, classifier or other parts.
  """
  def __init__(self, model, backbone, backbone_args, 
              classifier=None, classifier_args=None, 
              meta_args=None, mtd_args=None):

    # ====== Configure networks components ======
    # backbone
    self.backbone = backbones.make(backbone, **backbone_args)

    # classifier
    if model in st1_model:
        classifier_args['in_dim'] = self.backbone.out_dim
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_args["dropout_rate"]),
            models.make(classifier, **classifier_args),
            )
    # method
    if mtd_args:
      if mtd_args["attention"] is not None:
        if mtd_args["attention"] == "sot":
          #self.temp = nn.Parameter(torch.tensor(mtd_args["temp"]))
          self.mtd_opt = Attention()

  def get_backbone(self):
    return self.backbone

  def get_classifier(self):
    return self.classifier

  def get_feature_extractor(self):
    return self.feature_extractor

  def get_mtd_opt(self):
    return self.mtd_opt
