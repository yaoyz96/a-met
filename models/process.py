import numpy as np

import torch
import torch.nn.functional as F

"""
Make classifiction based on prototype.
"""
def classify(support_features: torch.Tensor, 
            support_labels: torch.Tensor, 
            query_features: torch.Tensor,
            metric: str = "cosine",
          ) -> torch.Tensor:
  """
  Return classification scores of query set in each task (with number of classes num_classes).
  :param support_features: (torch.tensor) Feature representation for each image in the support set (M x dim).
  :param support_labels: (torch.tensor) Labels for the support set (M x 1 -- integer representation).
  :param query_features: (torch.tensor) Feature representation for each image in the query set (N x dim).
  :return: (torch.tensor) Categorical distribution on label set for each image in query set (N x M).
  """
  class_representations = _build_class_prototype(support_features, support_labels)
  scores = similarity(class_representations, query_features, metric)

  # scores = similarity(support_features, query_features, metric)
  # scores = scores.view()

  return scores


def _build_class_prototype(support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
  """
  Construct and return class level representation (prototype) for each class in task.
  :param support_features: (torch.tensor) Feature representation for each image in the support set (M x dim).
  :param support_labels: (torch.tensor) Label for each image in the support set (M x 1).
  :return: (torch.tensor) the class representation tensor, ordered by class idx.
  """
  class_representations = torch.tensor([]).cuda(0)

  for c in torch.unique(support_labels):
    # filter out feature vectors which have class c
    class_features = torch.index_select(support_features, 0, _extract_class_indices(support_labels, c))
    class_rep = torch.mean(class_features, dim=0, keepdim=True)
    class_representations = torch.cat((class_representations, class_rep), dim=0)
  
  return class_representations


def _extract_class_indices(support_labels: torch.Tensor, class_idx: int) -> torch.Tensor:
  """
  Helper method to extract the indices of elements (::support_labels) which have the specified label (::class_idx).
  :param support_labels: (torch.tensor) Labels of the support set.
  :param class_idx: Label for which indices are extracted.
  :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
  """
  class_mask = torch.eq(support_labels, class_idx)  # binary mask of labels equal to which_class
  class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class

  return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def similarity(prototypes: torch.Tensor, query_features: torch.Tensor, metric: str ="cosine") -> torch.Tensor:
  """
  Helper function to calculate similarity scores between query set and support set.
  :param prototypes: (torch.tensor) class representation (prototype) of each class (num_classes x dim).
  :param query_features: (torch.tensor) features of query set (N x dim).
  :return: (torch.tensor) similarity scores (N x num_classes).
  """
  if metric == "cosine":
    scores = cosine_similarity(query_features, prototypes)
  elif metric == "eudlidean":
    scores = eudlidean_distance(query_features, prototypes)

  return scores


def inner_product(x, y, temp=1.0):
    '''
    Not recommond, no normalization of cosine similarity
    x: (BS x) N x FeatureDim
    y: (BS x) M x FeatureDim
    output: (BS x) N x M
    '''
    assert x.dim() == y.dim()

    if x.dim() == 2:
        score = torch.mm(x, y.t())
    elif x.dim() == 3:
        score = torch.bmm(x, y.permute(0, 2, 1))  

    return score * temp


def cosine_similarity(x, y, temp=1.0):
    '''
    Cosine Similarity, when n_shot = 1, recommend
    x: (BS x) N x FeatureDim
    y: (BS x) M x FeatureDim
    output: (BS x) N x M
    '''
    assert x.dim() == y.dim()

    if x.dim() == 2:
        score = torch.mm(F.normalize(x, dim=-1), F.normalize(y, dim=-1).t())
    elif x.dim() == 3:
        score = torch.bmm(F.normalize(x, dim=-1), F.normalize(y, dim=-1).permute(0, 2, 1))  

    return score * temp


def eudlidean_distance(x, y, temp=0.1):
    '''
    Eudlidean Distance, when n_shot > 1, recommend
    x: (BS x) N x FeatureDim
    y: (BS x) M x FeatureDim
    output: (BS x) N x M
    '''
    assert x.size(-1) == y.size(-1)

    if x.dim() == 2:
      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)
      
      dist = torch.pow(x - y, 2).sum(dim=-1)

    elif x.dim() == 3:
      b = x.size(0)
      n = x.size(1)
      m = y.size(1)
      d = x.size(2)

      x = x.unsqueeze(2).expand(b, n, m, d)
      y = y.unsqueeze(1).expand(b, n, m, d)
      
      dist = torch.pow(x - y, 2).sum(dim=-1)

    return dist * temp


'''post-process for episodic-training.
'''
class Similarity:
    def __init__(self, batch_size, metric, n_way, n_shot, n_query, temp=1.0):
        super(Similarity, self).__init__()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.batch_size = batch_size
        self.temp = temp
        if metric == 'inner_product':
            self.metric = inner_product
        elif metric == 'cosine_similarity':
            self.metric = cosine_similarity
        elif metric == 'eudlidean_distance':
            self.metric = eudlidean_distance

    def process(self, x):
        # create fake query label
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        y_query = y_query.repeat(self.batch_size)

        x = x.view(self.batch_size, self.n_way, self.n_shot + self.n_query, -1)
        fea_s = x[:, :, :self.n_shot]
        fea_q = x[:, :, self.n_shot:]
        proto = fea_s.contiguous().view(self.batch_size, self.n_way, self.n_shot, -1).mean(-2)  # [bs, n_way, *]
        fea_q = fea_q.contiguous().view(self.batch_size, self.n_way * self.n_query, -1)   # [bs, n_way*n_query, *]

        scores = self.metric(fea_q, proto, self.temp).view(-1, self.n_way)

        return scores, y_query
