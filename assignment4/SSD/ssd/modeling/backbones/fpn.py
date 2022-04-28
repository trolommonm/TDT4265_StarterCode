from collections import OrderedDict
from typing import List, Tuple
import torch.nn
from torch import nn
import torchvision

resnet_type_dict = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152
}


class FPN(nn.Module):
    """
    This is the backbone for the RetinaNet model for Task 2.3.
    """

    def __init__(self,
                 resnet_type,
                 output_channels: List[int],
                 output_feature_sizes: List[Tuple[int]],
                 pretrained=True):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.resnet_model = resnet_type_dict[resnet_type](pretrained=pretrained)

        # Get the output channels of the resnet model selected
        tmp = torch.rand((1, 3, 128, 1024))
        layers = list(self.resnet_model.children())[:-2]
        layers_output_channels = []
        for idx, l in enumerate(layers):
            tmp = l(tmp)
            if idx == 0 or idx == 1 or idx == 2 or idx == 3:
                continue
            layers_output_channels.append(tmp.shape[1])

        # We will be using layers 1 to 4 of the ResNet model for the first 4 feature extractors.
        # Define 2 more layers so that there are a total of 6 feature extractors
        self.layer5 = nn.Sequential(
            nn.Conv2d(layers_output_channels[-1], 512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, layers_output_channels[-1] * 2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(layers_output_channels[-1] * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(layers_output_channels[-1] * 2, 512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, layers_output_channels[-1] * 2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(layers_output_channels[-1] * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        layers_output_channels.extend([layers_output_channels[-1] * 2, layers_output_channels[-1] * 2])

        self.feature_extractors = nn.ModuleList([
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
            self.layer5,
            self.layer6
        ])

        self.fpn = torchvision.ops.FeaturePyramidNetwork(layers_output_channels, 256)

    def forward(self, x):
        """
        The forward function should output features with shape:
            [shape(-1, output_channels[0], 32, 256),
            shape(-1, output_channels[1], 16, 128),
            shape(-1, output_channels[2], 8, 64),
            shape(-1, output_channels[3], 4, 32),
            shape(-1, output_channels[3], 2, 16),
            shape(-1, output_channels[4], 1, 8)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 32, 256),
        """
        out_features = OrderedDict()

        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)

        for idx, feature_extractor in enumerate(self.feature_extractors):
            x = feature_extractor(x)
            out_features[f"feature_{idx}"] = x

        # for idx, feature in enumerate(out_features.values()):
        #     print(feature.shape[1:])

        output = self.fpn(out_features)

        for idx, feature in enumerate(output.values()):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape), \
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        return tuple(output.values())
