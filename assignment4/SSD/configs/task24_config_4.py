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
    min_sizes=[[ 14.43412597,  10.03293329],
            [ 31.69480147,  14.39732001],
            [ 48.74948972,  29.59106408],
            [ 41.87208617,  75.88263344],
            [ 68.3050001,  144.74697725],
            [126.60935507, 324.37612718],
            [290, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    # aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    aspect_ratios = [[1.12338274318344, 3.3657007511835078],
                     [1.5727034724121773, 4.3943944751770445],
                     [1.377497383890537, 3.978129662613269],
                     [0.43450315277367024, 0.8957057087959126],
                     [0.39511522570811425, 0.9296195630493029],
                     [0.357650781354406, 0.4965486797036837]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)
