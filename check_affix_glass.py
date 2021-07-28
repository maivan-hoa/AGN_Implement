# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 22:30:45 2021

@author: HoaMV
"""

import cv2
import numpy as np
import torch
from torchsummary import summary
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F


def affix_glass(path_image, path_glass):
    face = io.imread(path_image)
    face = face/255
    face = np.transpose(face,(2, 0, 1))
    face = torch.tensor(face)
    # print(face.shape)
    # plt.imshow(face)
    
    # face = faces[j, ...]
    # lấy kích thước kính và tọa độ cần chèn tương ứng với khuôn mặt j
    # 18	160	45	97	142	52
    
    glassWidth, glassHeight = 142, 52
    x1, x2, y1, y2 = 18, 160, 45, 97
    
    glass = io.imread(path_glass)
    glass = cv2.resize(glass, (160, 160), interpolation = cv2.INTER_AREA)
    glass = glass/255
    glass = np.transpose(glass, (2, 0, 1))
    glass = glass[:, 39:81, 21:138]
    # plt.imshow(glass)
    
    
    imgGlass = cv2.imread(r'C:\Users\DELL\Desktop\New folder\AGN-Implement\/data/glasses_mask.png', 0)
    r = 160 / imgGlass.shape[1]
    dim = (160, int(imgGlass.shape[0] * r))
    imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
    imgGlass = imgGlass[39:81, 21:138]
    
    imgGlass[imgGlass < 50] = 0
    imgGlass[imgGlass >= 50] = 255
    # plt.imshow(imgGlass)
    
    # Tạo mask là mảng nhị phân hai chiều
    mask_Glass = imgGlass/255 # kính trắng, nền đen
    
    glass = F.interpolate(torch.tensor(glass[None,...]), (glassHeight, glassWidth)) # resize theo batch, kích thước truyền vào phải là (B x C x H x W)
    mask = F.interpolate(torch.tensor(mask_Glass[None, None,...]), (glassHeight, glassWidth)) # None tự động thêm một chiều tương ứng
    # plt.imshow(glass[0].permute(1,2,0))
    
    # cắt vùng ảnh cần chứa kính trong khuôn mặt để xử lý
    roi = face[None, :, y1:y2, x1:x2]
    roi = roi - mask # mask có kính màu trắng (giá trị 1), nền màu đen (giá trị 0)
    roi = torch.clamp(roi, 0) # vị trí để đặt gọng kính sẽ có giá trị 0
    
    face[:, y1:y2, x1:x2] = glass[0] + roi[0] # glass đang có nền bằng 0 --> chèn gọng kính vào ảnh mặt
    
    return (face*255).long()



if __name__ == '__main__':
    path_image = r'C:\Users\DELL\Desktop\New folder\AGN-Implement\data\faces_me\hoa_57.png'
    path_glass = r"C:\Users\DELL\Desktop\New folder\AGN-Implement\data/eyeglasses/glasses000002-2.png"
    face = affix_glass(path_image, path_glass)
    
    plt.imshow(face.permute(1,2,0))
    

















