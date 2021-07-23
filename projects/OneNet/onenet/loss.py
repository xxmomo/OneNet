#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OneNet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, iou_rotate_calculate

from scipy.optimize import linear_sum_assignment
import cv2 
import numpy as np


class SetCriterion(nn.Module):
    """ This class computes the loss for OneNet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss_alpha = cfg.MODEL.OneNet.ALPHA
        self.focal_loss_gamma = cfg.MODEL.OneNet.GAMMA

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        bs, k, h, w = src_logits.shape
        src_logits = src_logits.permute(0, 2, 3, 1).reshape(bs, h * w, k)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_boxes
        losses = {'loss_ce': class_loss}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes']
        bs, k, h, w = src_boxes.shape
        src_boxes = src_boxes.permute(0, 2, 3, 1).reshape(bs, h * w, k)

        src_boxes = src_boxes[idx]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        #orgin giou
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes)) 
        #rotate iou
        loss_giou = 1 - torch.diag(box_ops.iou_rotate_calculate(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


class MinCostMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss_alpha = cfg.MODEL.OneNet.ALPHA
        self.focal_loss_gamma = cfg.MODEL.OneNet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, k, h, w = outputs["pred_logits"].shape
        print("outputs[pred_logits].shape:", outputs["pred_logits"].shape)
        print("outputs[pred_boxes].shape:", outputs["pred_boxes"].shape)
        # print("outputs:", outputs)

        # We flatten to compute the cost matrices in a batch

        batch_out_prob = outputs["pred_logits"].permute(0, 2, 3, 1).reshape(
            bs, h * w, k).sigmoid()  # [batch_size, num_queries, num_classes]
        batch_out_bbox = outputs["pred_boxes"].permute(0, 2, 3, 1).reshape(
            bs, h * w, 5)  # [batch_size, num_queries, 4] #把原来4改成了5

        indices = []

        for i in range(bs):
            tgt_ids = targets[i]["labels"] -1
            print("tgt_ids", tgt_ids)  # 形状是单个元素的tensor

            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]).to(batch_out_prob),
                               torch.as_tensor([]).to(batch_out_prob)))
                continue

            tgt_bbox = targets[i]["boxes_xyxy"] #最后一维是角度不是弧度
            print("tgt_bbox:", tgt_bbox)
            out_prob = batch_out_prob[i]
            out_bbox = batch_out_bbox[i]
            # print("out_bbox:", out_bbox)
            # print("out_bbox_shape:", out_bbox.shape)
            # print("tgt_bbox_shape:", tgt_bbox.shape) 
            
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            # unsqueeze(0)在位置0加一个维度，repeat分别在每个维度乘以相应倍数，第0维成w*h倍数
            image_size_out = targets[i]["image_size_xyxy"].unsqueeze(0).repeat(h * w, 1)
            # print("image_size_out:", image_size_out)
            # print("image_size_out_shape:", image_size_out.shape)
            image_size_tgt = targets[i]["image_size_xyxy_tgt"]
            # print("image_size_tgt:", image_size_out)
            # print("image_size_tgt_shape:", image_size_out.shape)
            print("out_bbox",out_bbox)
            out_bbox_ = out_bbox / image_size_out  # image_size_out是带角度的五个输出
            
            tgt_bbox_ = tgt_bbox / image_size_tgt
            cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

            # ##Compute the giou cost betwen boxes
            # origin giou
            # cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            # ##rotate boxes iou by xx
            # print('type(tgt_bbox):',type(tgt_bbox)) #torch.Tensor
            # tgt_bbox_trans =[]
            # out_bbox_trans=[]
            # tgt_bbox=tgt_bbox.cpu().numpy()
            # out_bbox=out_bbox.cpu().numpy()
            # for line in tgt_bbox:
            #     a=[[line[0],line[1]],[line[2],line[3]],line[4]]
            #     tgt_bbox_array=cv2.boxPoints(np.array(a))
            #     tgt_bbox_trans.append(tgt_bbox_array)
            # for line in out_bbox:
            #     b=[[line[0],line[1]],[line[2],line[3]],line[4]]
            #     out_bbox_array=cv2.boxPoints(np.array(b))
            #     out_bbox_trans.append(out_bbox_array)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print('tgt_bbox_trans:',tgt_bbox_trans)
            # # print('out_bbox_array:',out_bbox.shape)
            # cost_giou = -iou_rotate_calculate(np.array(out_bbox_trans), np.array(tgt_bbox_trans))
            cost_giou = -iou_rotate_calculate((out_bbox.cpu().numpy()), (tgt_bbox.cpu().numpy()))
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)
            indices.append((src_ind, tgt_ind))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
