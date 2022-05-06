from .task23_iteration4 import (
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
from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[ 10.00822415,  14.32669547],
        [ 14.33447421,  31.18747851],
        [ 28.59649771,  48.63004917],
        [ 71.90818914,  42.57914127],
        [134.64006415,  59.252283  ],
        [289.38266839, 126.09288907],
        [290, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    # aspect_ratios = [[0.84756, 2, 3], [3.2, 2, 1.27505], [3, 4.20657, 2.70044], [2.43431, 3, 4.12147], [0.69223, 1.41207, 0.29947], [0.8378, 0.43759, 0.39032]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)
