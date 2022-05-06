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

# cluster based on area, sqrt the area to get min_sizes
# cluster the points on each area to get aspect ratios

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[ 16.50011045,  16.50011045],
            [ 40.85609875,  40.85609875],
            [ 62.50417139,  62.50417139],
            [ 90.04720867,  90.04720867],
            [157.34485365, 157.34485365],
            [212.42028297, 212.42028297],
            [220, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    # aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    aspect_ratios = [[1.6260005787570107, 2.248044383734856],
                     [1.4136707838678264, 2.078552274637193],
                     [1.183178146453538, 2.691955864976278],
                     [1.15682904214359, 2.596731274934622],
                     [1.2582115246224301, 1.7668584307390793],
                     [2.596770784372394, 3.0215736751059112]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)
