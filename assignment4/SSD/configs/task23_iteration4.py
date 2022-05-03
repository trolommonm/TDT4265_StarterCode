from .task23_iteration3 import (
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
import torch

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    use_improved_weight=True,
    use_deeper_heads=True
)

loss_objective.alpha = torch.FloatTensor([100, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])