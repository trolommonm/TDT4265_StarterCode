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
from tops.config import LazyCall as L

backbone = L(backbones.FPN)(
    resnet_type="resnet34",
    pretrained=True
)
