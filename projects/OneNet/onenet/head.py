#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OneNet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .deconv import CenternetDeconv


class Head(nn.Module):

    def __init__(self, cfg, backbone_shape=[2048, 1024, 512, 256]):
        super().__init__()

        # Build heads.
        num_classes = cfg.MODEL.OneNet.NUM_CLASSES
        d_model = cfg.MODEL.OneNet.DECONV_CHANNEL[-1]
        activation = cfg.MODEL.OneNet.ACTIVATION

        self.deconv = CenternetDeconv(cfg, backbone_shape)

        self.num_classes = num_classes
        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = _get_activation_fn(activation)

        self.feat1 = nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1)
        self.cls_score = nn.Conv2d(d_model, num_classes, kernel_size=3, stride=1, padding=1)
        self.ltrb_pred = nn.Conv2d(d_model, 4, kernel_size=3, stride=1, padding=1)
        self.pre_angle = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.angle_pred = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)  # xx加
        # Init parameters.
        prior_prob = cfg.MODEL.OneNet.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)

    def forward(self, features_list):

        features = self.deconv(features_list)
        locations = self.locations(features)[None]

        feat = self.activation(self.feat1(features))

        class_logits = self.cls_score(feat)
        pred_ltrb = F.relu(self.ltrb_pred(feat))  # 3x3卷积加relu激活
        # print("pred_ltrb:", pred_ltrb)
        # print("pred_ltrb.shape:", pred_ltrb.shape)
        pred_angle = F.relu(self.pre_angle(feat))
        pred_angle = self.angle_pred(pred_angle)
        pred_bboxes = self.apply_ltrb(locations, pred_ltrb, pred_angle)
        pred_bboxes = torch.cat((pred_bboxes, pred_angle), 1)  # xx加
        print("pred_bboxes.shape:", pred_bboxes.shape)

        return class_logits, pred_bboxes

    # def apply_ltrb(self, locations, pred_ltrb):
    #     """
    #     :param locations:  (1, 2, H, W)
    #     # :param pred_ltrb:  (N, 4, H, W)
    #     """

    #     pred_boxes = torch.zeros_like(pred_ltrb)
    #     pred_boxes[:,0,:,:] = locations[:,0,:,:] - pred_ltrb[:,0,:,:]  # x1
    #     pred_boxes[:,1,:,:] = locations[:,1,:,:] - pred_ltrb[:,1,:,:]  # y1
    #     pred_boxes[:,2,:,:] = locations[:,0,:,:] + pred_ltrb[:,2,:,:]  # x2
    #     pred_boxes[:,3,:,:] = locations[:,1,:,:] + pred_ltrb[:,3,:,:]  # y2
    #     # pred_boxes[:,4,:,:] = pred_ltrb[:,4,:,:]  # theta #这样就可以算出theta？？？？

    #     return pred_boxes
    def apply_ltrb(self, locations, pred_ltrb, pred_angle):
        """
        :param locations:  (1, 2, H, W)
        # :param pred_ltrb:  (N, 4, H, W)   # :param pred_ltrb:  (N, 8, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)

        print("pred_angle:", pred_angle.shape)
        pred_angle = pred_angle[:, 0, :, :]
        # a = torch.div(pred_ltrb[:, 1, :, :], pred_ltrb[:, 1, :, :]
        # print("a", a.shape)
        
        # angle_alpha1 = torch.atan(pred_ltrb[:, 1, :, :] / pred_ltrb[:, 0, :, :])
        # angle_trans1 = torch.sub(angle_alpha1, pred_angle)  # 弧度还是角度？三角函数都是弧度
        # # print('angle_trans1.shape', angle_trans1.shape)
        # length1 = torch.sqrt(torch.pow(pred_ltrb[:, 0, :, :], 2)
        #                      + torch.pow(pred_ltrb[:, 1, :, :], 2))  # 平方和开根号
        # print('length1.shape', length1.shape, length1.size())
        # x1 = length1 * torch.cos(angle_trans1)  # 注意
        # y1 = length1 * torch.sin(angle_trans1)

        # angle_alpha2 = torch.atan(torch.div(pred_boxes[:, 1, :, :], pred_boxes[:, 2, :, :]))
        # angle_trans2 = torch.sub(angle_alpha2, pred_angle)  # 弧度还是角度？三角函数都是弧度
        # length2 = torch.sqrt(torch.pow(pred_ltrb[:, 2, :, :], 2)
        #                      + torch.pow(pred_ltrb[:, 1, :, :], 2))
        # x2 = length2 * torch.cos(angle_trans2)
        # y2 = length2 * torch.sin(angle_trans2)

        # angle_alpha3 = torch.atan(torch.div(pred_boxes[:, 3, :, :], pred_boxes[:, 2, :, :]))
        # angle_trans3 = torch.add(angle_alpha3, pred_angle)  # 弧度还是角度？三角函数都是弧度
        # length3 = torch.sqrt(torch.pow(pred_ltrb[:, 2, :, :], 2)
        #                      + torch.pow(pred_ltrb[:, 3, :, :], 2))
        # x3 = length3 * torch.cos(angle_trans3)
        # y3 = length3 * torch.sin(angle_trans3)

        # angle_alpha4 = torch.atan(torch.div(pred_boxes[:, 3, :, :], pred_boxes[:, 0, :, :]))
        # angle_trans4 = torch.add(angle_alpha4, pred_angle)  # 弧度还是角度？三角函数都是弧度
        # length4 = torch.sqrt(torch.pow(pred_ltrb[:, 0, :, :], 2)
        #                      + torch.pow(pred_ltrb[:, 3, :, :], 2))
        # x4 = length4 * torch.cos(angle_trans4)
        # y4 = length4 * torch.sin(angle_trans4)
        # pred_boxes[:, 0, :, :] = torch.div(x1 + x2 + x3 + x4, 4)
        # pred_boxes[:, 1, :, :] = torch.div(y1 + y2 + y3 + y4, 4)
        # pred_boxes[:, 2, :, :]=torch.add(pred_ltrb[:, 0, :, :],pred_ltrb[:, 2, :, :])
        # pred_boxes[:, 3, :, :]=torch.add(pred_ltrb[:, 1, :, :],pred_ltrb[:, 3, :, :])



        xc = torch.div(pred_ltrb[:, 0, :, :] + pred_ltrb[:, 2, :, :], 2)
        # print('xc.shape:', xc.shape)
        yc = torch.div(pred_ltrb[:, 1, :, :] + pred_ltrb[:, 3, :, :], 2)
        x1 = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        # print('x1.shape:', x1.shape)
        y1 = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        x2 = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        y2 = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        x3 = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        y3 = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2
        x4 = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        y4 = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2
        # pred_angle = pred_angle[:, 0, :, :]

        x1_ro = xc + (x1 - xc) * torch.cos(pred_angle) + \
            (y1 - yc) * torch.sin(pred_angle)
        y1_ro = yc - (x1 - xc) * torch.sin(pred_angle) + \
            (y1 - yc) * torch.cos(pred_angle)

        x3_ro = xc + (x3 - xc) * torch.cos(pred_angle) + \
            (y3 - yc) * torch.sin(pred_angle)
        y3_ro = yc - (x3 - xc) * torch.sin(pred_angle) + \
            (y3 - yc) * torch.cos(pred_angle)
        pred_boxes[:, 0, :, :] = torch.div(torch.add(x1_ro, x3_ro), 2)
        pred_boxes[:, 1, :, :] = torch.div(torch.add(y1_ro, y3_ro), 2)
        pred_boxes[:, 2, :, :] = torch.add(pred_ltrb[:, 0, :, :],pred_ltrb[:, 2, :, :])
        pred_boxes[:, 3, :, :] = torch.add(pred_ltrb[:, 1, :, :],pred_ltrb[:, 3, :, :])

        return pred_boxes

    @ torch.no_grad()
    def locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W, θ)  #原features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]  # 是否需要根据角度来换wh位置??????feature是否包含角度--否
        device = features.device

        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
