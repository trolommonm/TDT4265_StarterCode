from collections import OrderedDict
from typing import List
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
                 pretrained=True):
        super().__init__()
        # self.out_channels = [64, 128, 256, 512, 1024, 2048]
        # self.out_channels = [256, 256, 256, 256, 256, 256]
        self.out_channels = [64, 64, 64, 64, 64, 64]
        self.resnet_model = resnet_type_dict[resnet_type](pretrained=pretrained)

        # We will be using layers 1 to 4 of the ResNet model for the first 4 feature extractors.
        # Define 2 more layers so that there are a total of 6 feature extractors
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.feature_extractors = nn.ModuleList([
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
            self.layer5,
            self.layer6
        ])

        self.fpn = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 1024, 2048], 64)

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

        return tuple(output.values())
