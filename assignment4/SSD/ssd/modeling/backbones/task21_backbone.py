from torch import nn
from typing import Tuple, List


class BaselineModel(nn.Module):
    """
    This is a baseline model for Task 2.1, implemented using the tips provided
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        feature_extractor1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=output_channels[0],
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        feature_extractor2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[1],
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        feature_extractor3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[2],
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        feature_extractor4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[3],
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        feature_extractor5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[4],
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        feature_extractor6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=128,
                kernel_size=(2, 2),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[5],
                kernel_size=(2, 2),
                stride=2,
                padding=0
            ),
            nn.ReLU()
        )
        self.feature_extractors = nn.ModuleList([
            feature_extractor1,
            feature_extractor2,
            feature_extractor3,
            feature_extractor4,
            feature_extractor5,
            feature_extractor6
        ])

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
        out_features = []
        for feature_extractor in self.feature_extractors:
            x = feature_extractor(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        return tuple(out_features)

