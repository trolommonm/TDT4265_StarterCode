import torch
import torch.nn as nn
import numpy as np
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


def subnet(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    )


class RetinaNet(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 anchors,
                 loss_objective,
                 num_classes: int,
                 use_improved_weight=False,
                 use_deeper_heads=False):
        super().__init__()
        """
            Implements the RetinaNet network.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.regression_heads = []
        self.classification_heads = []
        self.anchors = anchors
        self.num_anchors = anchors.num_boxes_per_fmap[0]
        self.use_deeper_heads = use_deeper_heads

        if use_deeper_heads:
            # task 2.3.3 use shared deeper regression and classification heads
            in_channels = self.feature_extractor.out_channels[0]
            self.regression_heads = subnet(in_channels, self.num_anchors * 4)
            self.classification_heads = subnet(in_channels, self.num_anchors * self.num_classes)
        else:
            # Initialize output heads that are applied to each feature map from the backbone.
            for n_boxes, out_ch in zip(anchors.num_boxes_per_fmap, self.feature_extractor.out_channels):
                self.regression_heads.append(nn.Conv2d(out_ch, n_boxes * 4, kernel_size=3, padding=1))
                self.classification_heads.append(nn.Conv2d(out_ch, n_boxes * self.num_classes, kernel_size=3, padding=1))

                self.regression_heads = nn.ModuleList(self.regression_heads)
                self.classification_heads = nn.ModuleList(self.classification_heads)

        self.anchor_encoder = AnchorEncoder(anchors)

        self._init_weights(use_improved_weight)

    def _init_weights(self, use_improved_weight):
        layers = [*self.regression_heads, *self.classification_heads]

        if use_improved_weight:
            # task 2.3.4 weight initialization
            for layer in layers:
                if hasattr(layer, "bias"):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

            p = 0.99
            b = np.log(p * (self.num_classes - 1) / (1 - p))
            nn.init.constant_(self.classification_heads[-1].bias[:self.num_anchors], b)
        else:
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for idx, x in enumerate(features):
            if self.use_deeper_heads:
                bbox_delta = self.regression_heads(x).view(x.shape[0], 4, -1)
                bbox_conf = self.classification_heads(x).view(x.shape[0], self.num_classes, -1)
            else:
                bbox_delta = self.regression_heads[idx](x).view(x.shape[0], 4, -1)
                bbox_conf = self.classification_heads[idx](x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)

    def forward_test(self,
                     img: torch.Tensor,
                     imshape=None,
                     nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions


def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
    """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
    assert 0 <= nms_iou_threshold <= 1
    assert max_output > 0
    assert 0 <= score_threshold <= 1
    scores, category = confs.max(dim=1)

    # 1. Remove low confidence boxes / background boxes
    mask = (scores > score_threshold).logical_and(category != 0)
    boxes_ltrb = boxes_ltrb[mask]
    scores = scores[mask]
    category = category[mask]

    # 2. Perform non-maximum-suppression
    keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

    # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
    keep_idx = keep_idx[:max_output]
    return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]
