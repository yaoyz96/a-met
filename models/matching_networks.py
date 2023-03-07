import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as f

from .models import register

@register('matching_networks')
class MatchingNet(nn.Module):
    """
    This is the Matching Network which first creates embeddings for all images in the
    support sets and the target sets. Then it uses those embeddings to use the attentional
    classifier
    """
    def __init__(self, num_classes, input_channels=1):
        super(MatchingNet, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        self.embed = EmbedNet(self.input_channels)
        self.attend = AttentionNet()

    def forward(self, support_sets, support_set_labels, target_images):
        """
        :param support_sets: [batch_size, k, H, W] dimensional tensor, contains support image sets
        :param support_set_labels: [batch_size, k, 1] dimensional tensor, contains support image labels
        :param target_images: [batch_size, H, W]
        :return:
        """
        batch_size, k, h, w = support_sets.size()
        support_embeddings = self.embed(support_sets.view(batch_size*k, h, w)).view(batch_size, k, -1)
        target_embeddings = self.embed(target_images)

        support_set_one_hot_labels = torch.FloatTensor(batch_size, k, self.num_classes).zero_()
        support_set_one_hot_labels.scatter_(2, support_set_labels, 1)

        attention_classify = self.attend(support_embeddings, support_set_one_hot_labels, target_embeddings)
        return attention_classify


class EmbedNet(nn.Module):
    """
    This class implements the embedding CNN (referenced as f' in the literature)
    It takes in a 1x28x28 image as input and returns a 64x1 vector for each image.
    This happens via stacking 4 modules of:
        64 (3x3) Convolution Filters -> Batch Normalization -> ReLU -> Subsampling [-> Dropout]
    All the parameters are initialized via Glorot initialization
    """
    def __init__(self, input_channels, dropout_prob=0.1):
        super(EmbedNet, self).__init__()

        self.dropout_prob = dropout_prob

        self.module1 = self._create_module(input_channels, 64)
        self._init_weights(self.module1)

        self.module2 = self._create_module(64, 64)
        self._init_weights(self.module2)

        self.module3 = self._create_module(64, 64)
        self._init_weights(self.module3)

        self.module4 = self._create_module(64, 64)
        self._init_weights(self.module4)

    def forward(self, images):
        output = self.module1(images)
        output = self.module2(output)
        output = self.module3(output)
        output = self.module4(output)

        batch_size, *dims = output.size()
        flatten_size = reduce(lambda x,y: x*y, dims)

        output = output.view(batch_size, 1, flatten_size)
        return output

    def _create_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout2d(self.dropout_prob)
        )

    def _init_weights(self, mod, nonlinearity='relu'):
        for m in mod:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight, gain=nn.init.calculate_gain(nonlinearity))
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


class AttentionNet(nn.Module):
    """
    This network computes the attention - softmax over cosine similarities between the support
    set and a target image. The output is the label computed as the max pdf over
    all the labels in the support set (referenced as y_hat in the literature)
    The inputs in the forward pass are the embeddings computed previously
    (referenced as f and g in the literature)
    """
    def __init__(self):
        super(AttentionNet, self).__init__()

        self.layer = nn.Softmax()

    def forward(self, support_sets, support_set_labels, target_images):
        """
        Compute the forward pass
        :param support_sets: [batch_size, k, D] dimensional tensors
        :param support_set_labels: [batch_size, k, num_classes] dimensional tensors, one-hot encoded labels
        :param target_images: [batch_size, D] dimensional tensors
        :return: [batch_size, 1] dimensional tensors as the pdf for target images
        """

        # Compute the cosine similarity by dot product of two L2 normalized vectors
        support_sets_norm = f.normalize(support_sets, p=2, dim=2)
        target_images_norm = f.normalize(target_images, p=2, dim=1).unsqueeze(dim=1).permute(0, 2, 1)
        similarities = torch.bmm(support_sets_norm, target_images_norm).squeeze(dim=2)

        # Compute the softmax and distribution over all classes
        softmax_pdf = self.layer(similarities).unsqueeze(dim=1)
        prediction_pdf = torch.bmm(softmax_pdf, support_set_labels)

        _, predictions = torch.max(prediction_pdf, dim=2)
        return predictions