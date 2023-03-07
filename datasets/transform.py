import torch
import torch.nn as nn

import torchvision.transforms as transforms


class TransformLoader:
    def __init__(self, image_size):
        super(TransformLoader, self).__init__()

        self.normalize_param = dict(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        self.image_size = image_size
        if image_size == 28:
            self.resize_size = 28
        elif image_size == 84:
            self.resize_size = 92
        elif image_size == 224:
            self.resize_size = 256

    def getTransform(self, aug=False):
        if aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
            
        return transform