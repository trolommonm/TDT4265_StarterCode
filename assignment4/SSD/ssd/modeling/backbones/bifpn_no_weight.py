from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from typing import List
from tops.torch_utils import to_cuda


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.


    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_td = DepthwiseConvBlock(feature_size, feature_size)

        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p8_out = DepthwiseConvBlock(feature_size, feature_size)

        # TODO: Init weights
#         self.w1 = nn.Parameter(torch.Tensor(2, 5))
#         self.w1_relu = nn.ReLU()
#         self.w2 = nn.Parameter(torch.Tensor(3, 5))
#         self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x, p8_x = inputs

        # Calculate Top-Down Pathway
#         w1 = self.w1_relu(self.w1)
#         w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
#         w2 = self.w2_relu(self.w2)
#         w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        p8_td = p8_x
        p7_td = self.p7_td(p7_x + F.interpolate(p8_td, scale_factor=2))
        p6_td = self.p6_td(p6_x + F.interpolate(p7_td, scale_factor=2))
        p5_td = self.p5_td(p5_x + F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(p4_x + F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(p3_x + F.interpolate(p4_td, scale_factor=2))

        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(p4_x + p4_td + nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(p5_x + p5_td + nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(p6_x + p6_td + nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(p7_x + p7_td + nn.Upsample(scale_factor=0.5)(p6_out))
        p8_out = self.p8_out(p8_x + p8_td + nn.Upsample(scale_factor=0.5)(p7_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]


class BiFPN(nn.Module):

    def __init__(self,
                output_channels: List[int],
                feature_size=64):
        super(BiFPN, self).__init__()
        self.out_channels = [feature_size] * 6
        self.resnet_model = torchvision.models.resnet34(pretrained=True)

        self.p0 = nn.Conv2d(output_channels[0], feature_size, 1)
        self.p1 = nn.Conv2d(output_channels[1], feature_size, 1)
        self.p2 = nn.Conv2d(output_channels[2], feature_size, 1)
        self.p3 = nn.Conv2d(output_channels[3], feature_size, 1)
        self.p4 = nn.Conv2d(output_channels[4], feature_size, 1)
        self.p5 = nn.Conv2d(output_channels[5], feature_size, 1)

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1),
                      padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(2048),
            nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1),
                      padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        self.feature_extractors = nn.ModuleList([
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
            self.layer5,
            self.layer6
        ])

        self.bifpn = nn.Sequential(BiFPNBlock(feature_size), BiFPNBlock(feature_size))

        # [64, 128, 256, 512, 1024, 2048]

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)

        out_features = OrderedDict()

        for idx, feature_extractor in enumerate(self.feature_extractors):
            x = feature_extractor(x)
            out_features[f"feature_{idx}"] = x

        x0, x1, x2, x3, x4, x5 = out_features.values()

        x0 = self.p0(x0)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x4 = self.p4(x4)
        x5 = self.p5(x5)

        features = [x0, x1, x2, x3, x4, x5]

        return self.bifpn(features)
