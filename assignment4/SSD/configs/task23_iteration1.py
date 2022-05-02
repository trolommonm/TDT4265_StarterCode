from .task21_config import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)
from ssd.modeling import backbones
from ssd.modeling import RetinaNet
from tops.config import LazyCall as L

backbone = L(backbones.FPN)(
    resnet_type="resnet34",
    output_channels=[256, 256, 256, 256, 256, 256],
    output_feature_sizes="${anchors.feature_sizes}",
    pretrained=True
)

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1  # Add 1 for background
)