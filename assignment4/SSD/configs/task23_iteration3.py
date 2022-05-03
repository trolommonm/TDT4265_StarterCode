from .task23_iteration2 import (
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

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    use_improved_weight=False,
    use_deeper_heads=True
)

anchors.aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
