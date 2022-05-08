from .task25_config_2 import (
    train,
    optimizer,
    schedulers,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors,
    backbone,
    model,
    loss_objective
)
from ssd.modeling import RetinaNet
from tops.config import LazyCall as L
from ssd.modeling.backbones import BiFPN

backbone = L(BiFPN)(
    output_channels=[64, 128, 256, 512, 1024, 2048],
    feature_size=256
)