import numpy as np

import torch
from torch.utils.data import Sampler


"""
Sampling data by classes index.
"""
class EpisodicSampler(Sampler):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


"""
Sampling data by classes index with batch size > 1.
"""
class EpisodicBatchSampler():
    def __init__(self, n_classes, n_way, n_episodes, batch_size=1):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.batch_size = batch_size

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            batch = []
            for b in range(self.batch_size):
                classes = torch.randperm(self.n_classes)[:self.n_way]
                batch.append(classes)
            batch = torch.stack(batch)
            yield batch.view(-1)


