# Inherit all the configs from Task 2.1
from .task21_config import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)
from ssd.data.transforms import ToTensor, Normalize, Resize, GroundTruthBoxesToAnchors
import torchvision
from tops.config import LazyCall as L
from ssd.data.transforms import RandomSampleCrop, RandomHorizontalFlip


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(RandomHorizontalFlip)(),
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])


