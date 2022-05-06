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
    min_sizes=[[14.84488255, 10.21969198],
               [33.64265557, 14.38769218],
               [48.68630141197569, 31.60240590197752],
               [41.93129823223988, 73.03535251746572],
               [58.23934642318025, 132.45558824726197],
               [123.10415116795959, 276.3839632430168],
               [290, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    # aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    aspect_ratios = [[2.6881450129420714, 1.1038069304264213],
                     [0.42612519737811105, 1.2997929631222307],
                     [1.8017118148716196, 0.7796891625103304],
                     [1.0274373813359166, 1.9618092522560668],
                     [1.0295293982898177, 0.4190230557037386],
                     [1.606943055514171, 1.0555325788133025]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)
