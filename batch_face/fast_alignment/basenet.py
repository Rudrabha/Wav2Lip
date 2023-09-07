# Backbone networks used for face landmark detection
# Cunjian Chen (cunjian@msu.edu)

import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


# SE module
# https://github.com/wujiyang/Face_Pytorch/blob/master/backbone/cbam.py
class SEModule(nn.Module):
    """Squeeze and Excitation Module"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x


# USE global depthwise convolution layer. Compatible with MobileNetV2 (224×224), MobileNetV2_ExternalData (224×224)
class MobileNet_GDConv(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet_GDConv, self).__init__()
        self.pretrain_net = models.mobilenet_v2()
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)

    def forward(self, x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


# USE global depthwise convolution layer. Compatible with MobileNetV2 (56×56)
class MobileNet_GDConv_56(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet_GDConv_56, self).__init__()
        self.pretrain_net = models.mobilenet_v2()
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (2, 2), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)

    def forward(self, x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


# MobileNetV2 with SE; Compatible with MobileNetV2_SE (224×224) and MobileNetV2_SE_RE (224×224)
class MobileNet_GDConv_SE(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet_GDConv_SE, self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=True)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
        self.attention = SEModule(1280, 8)

    def forward(self, x):
        x = self.base_net(x)
        x = self.attention(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x
