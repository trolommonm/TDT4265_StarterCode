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
from .utils import get_dataset_dir
from ssd.modeling import AnchorBoxes
from tops.config import LazyCall as L

data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[ 18.85107333,  18.85107333],
            [ 50.49308831,  50.49308831],
            [ 86.26703783,  86.26703783],
            [132.04802891, 132.04802891],
            [207.90202832, 207.90202832],
            [271.91442122, 271.91442122],
            [300, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    # aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    aspect_ratios = [[1.4461662615950077, 1.9982719286857764],
                     [1.4287885943406409, 1.8557411225185128],
                     [1.0650018341985334, 2.612718213953719],
                     [1.3775835346770167, 3.4314368808160856],
                     [2.4187324479976535, 3.4293772786574683],
                     [4.111277280369799, 5.5272120546062835]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)