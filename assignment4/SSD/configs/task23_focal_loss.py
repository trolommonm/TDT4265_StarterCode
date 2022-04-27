from .task23_config import (
    train,
    optimizer,
    schedulers,
    model,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors,
    backbone
)
from ssd.modeling import FocalLoss
from tops.config import LazyCall as L

loss_objective = L(FocalLoss)(anchors="${anchors}")
