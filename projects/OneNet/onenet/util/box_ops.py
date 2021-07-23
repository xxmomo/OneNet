# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import cv2
import numpy as np
# from torch._C import float32
# from torch._C import R
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# def box_xyxy_to_cxcywh(x):
#     print("x",x.shape)
#     # y = x[:, 0:4]
#     x0, y0, x1, y1,theta = x.unbind(-1)
#     # 这里的x0，x1.。。是不是真的可以代表两个点，
#     # 是不是他们求均值和相减就能得到x,y,w,h????????
#     b = [(x0 + x1) / 2, (y0 + y1) / 2
#          (x1 - x0), (y1 - y0),theta]
#     return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywhtheta(x):
    # print("x——shape", x.shape)
    # print('x',x)
    # y = x[:, 0:4]
    x0, y0, x1, y1, theta = x.unbind(-1)
    # 这里的x0，x1.。。是不是真的可以代表两个点，
    # 是不是他们求均值和相减就能得到x,y,w,h????????
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    b = [cx, cy, x1 - x0, y1 - y0, theta]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

# 旋转iou


def iou_rotate_calculate(boxes1, boxes2):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []      
        # opencv必须读整型数据，要把r1，r2变成整型\
        r1 = [[int(box1[0]), int(box1[1])], [int(box1[2]), int(box1[3])],
              int(box1[4])]  # x,y,w,h,theta
        
        # r2 = [[box2[0][0].astype(np.int), box2[0][1].astype(np.int)], [box2[1][0].astype(np.int), box2[1][1].astype(np.int)], [
        #     box2[2][0].astype(np.int), box2[2][1].astype(np.int)], [box2[3][0].astype(np.int), box2[3][1].astype(np.int)]]  # 四个顶点坐标

        for j, box2 in enumerate(boxes2):
            # print('box1', box1)
            r2 = [[int(box2[0]), int(box2[1])], [int(box2[2]), int(box2[3])],
              int(box2[4])]  # X,Y,W,H,THETA
            # r1 = [[box1[0][0].astype(np.int), box1[0][1].astype(np.int)], [box1[1][0].astype(np.int), box1[1][1].astype(np.int)], [
            #     box1[2][0].astype(np.int), box1[2][1].astype(np.int)], [box1[3][0].astype(np.int), box1[3][1].astype(np.int)]]  # 四个顶点坐标
            # 这个函数计算两个旋转矩形的交集，返回值维0，1，2分别表示没有，有，包含；
            # 以及交点的坐标的一个array很遗憾的是这个坐标是不是逆时针也不是顺时针。
            # 而cv2.contourArea()需要点是顺时针或者逆时针，解决办法？？？

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]  # 获得交点坐标的array
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)  # 返回凸包上点的坐标
                # 计算轮廓的面积，就是算包括boxes1 和boxes2最小外包面积
                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return torch.tensor(ious, dtype=torch.float32).to(device)
    # return np.array(ious, dtype=np.float32)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
