import torch
import torch.nn as nn

from .backbones import register


################################################# 
#           Simple Variant of ResNet            #
#################################################

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)

"""Variant of the original BasicBlock
"""
class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, downsample):
        super().__init__()

        self.conv1 = conv3x3(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv3x3(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)

        self.relu = nn.LeakyReLU(0.1, inplace=False)   # LeakyReLU
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x.contiguous())

        out += identity
        out = self.relu(out)
        out = self.maxpool(out)

        return out.contiguous()


class ResNet12(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.gradls = []

        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = BasicBlock(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        
        return x


@register('ResNet12')
def resnet12():
    return ResNet12([64, 128, 256, 512])


@register('ResNet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])

