import torch.nn as nn
import torch
import torch.nn.functional as F
from tops import to_cuda


def focal_loss(confs, gt_labels):
    """
    confs: [batch_size, num_classes, num_anchors]
    gt_labels = [batch_size, num_anchors]
    """
    num_classes = 8 + 1  # add 1 for background class
    num_anchors = gt_labels.shape[1]
    batch_size = gt_labels.shape[0]
    gt_labels = torch.permute(F.one_hot(gt_labels, num_classes), (0, 2, 1))  # [batch_size, num_classes, num_anchors]

    soft_confs = F.softmax(confs, dim=1)
    log_soft_confs = F.log_softmax(confs, dim=1)

    # alpha = torch.FloatTensor([0.01, 1, 1, 1, 1, 1, 1, 1, 1])
    alpha = torch.FloatTensor([10, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    alpha = alpha.view(1, -1, 1)
    alpha = to_cuda(alpha)
    gamma = 2
    x = - alpha * (1 - soft_confs) ** gamma * gt_labels * log_soft_confs  # [batch_size, num_classes, num_anchors]
    # x = -torch.sum(x, axis=2)  # [batch_size, num_anchors]
    # x = torch.sum(x, axis=1) / num_anchors  # [batch_size]
    # x = torch.sum(x)  # / batch_size
    x = x.sum(dim=1).mean()

    return x


class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0 / anchors.scale_xy
        self.scale_wh = 1.0 / anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                    requires_grad=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.anchors[:, :2, :]) / self.anchors[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self,
                bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
                gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous()  # reshape to [batch_size, 4, num_anchors]

        """
        with torch.no_grad():
            to_log = - F.log_softmax(confs, dim=1)[:, 0]
            mask = hard_negative_mining(to_log, gt_labels, 3.0)
        classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        classification_loss = classification_loss[mask].sum()
        """

        # focal loss
        classification_loss = focal_loss(confs, gt_labels)
        print("classification_loss: ", classification_loss)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0] / 4
        total_loss = regression_loss / num_pos + classification_loss  # / num_pos
        to_log = dict(
            regression_loss=regression_loss / num_pos,
            classification_loss=classification_loss,  # / num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log
