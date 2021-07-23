import torch
import numpy as np
import cv2
# def box_xyxy_to_cxcywh(x):
#     y = x[0:3]
#     x0, y0, x1, y1 = y.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)
# x= torch.ones(1, 5)
# print("x:", x)
# y = x[:,0:4]
# print("y:", y)
# z= box_xyxy_to_cxcywh(y)
# print("z:", z)

# stride = 4
# w = 10
# h = 10
# shifts_x = torch.arange(
#     0, w * stride, step=stride,
#     dtype=torch.float32
# )
# print("shifts_x", shifts_x)
# shifts_y = torch.arange(
#     0, h * stride, step=stride,
#     dtype=torch.float32
# )
# shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
# print("shift_x.meshgrid:", shift_x)
# shift_x = shift_x.reshape(-1)
# print("shift_x.reshape:", shift_x)
# shift_y = shift_y.reshape(-1)
# locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
# print(locations)
# print(locations.shape)

# locations = locations.reshape(h, w, 2).permute(2, 0, 1)
# print(locations)
# r1 = [[1.1, 1.1], [2, 2], [3, 3]]
# temp1 = [ [int(i[0]),int(i[1])] for i in r1]
# print(temp1)

# pred_ltrb = torch.ones(8,4,2,2)
# print(pred_ltrb)
# pred_boxes = torch.zeros_like(pred_ltrb)
# print(pred_boxes.shape)
# pred_boxes = pred_boxes.repeat([1, 2, 1, 1])
# print(pred_boxes.shape)

x = np.array([[546.448, 136.192, 100.7859, 30.172728, -18.778032],
[1, 2, 19, 38, -132]])
# x=np.array([[1.,2],[3,4],[5,6],[7,8]])
# xx=cv2.minAreaRect(x)
# print('xx',xx)
result=[]
for line in x:
    b=((line[0],line[1]),(line[2],line[3]),line[4])
    result.append(b)

# y = cv2.boxPoints(x)
print(result)
