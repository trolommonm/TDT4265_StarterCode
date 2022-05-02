from .task23_iteration1 import (
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
    model
)
from ssd.modeling import FocalLoss
from tops.config import LazyCall as L

loss_objective = L(FocalLoss)(anchors="${anchors}")
