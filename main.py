# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:30:26 2021

@author: Mai Van Hoa - HUST
"""

from imports import *
from architecture import *
from model_inference import *
from dataset import *
from training_AGN import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CLASSES = {
    0: 'Bien',
    1: 'Cuong',
    2: 'LeDong',
    3: 'Phu',
    4: 'Vu',
    5: 'Nguyen',
    6: 'Hoa'
}

batch_size = 64
num_epochs = 10
class_names = ['Bien', 'Cuong', 'LeDong', 'Phu', 'Vu', 'Nguyen', 'Hoa']

nc, ndf, ngf, nz = 3, 160, 160, 100 # nz là chiều dài vector nhiễu đầu vào

# Khởi tạp Generator và Discriminator, Model Face Classification
netG = Generator(nc=nc, ngf=ngf, nz=nz)
netD = Discriminator(nc=nc, ndf=ndf)

path_model_classification = './model/MobileFaceNet_classification.pth'
model_clf = get_MobileFaceNet_classification(path_model_classification)

# Xây dựng DataLoader cho kính và khuôn mặt
eyeglassesDataset = EyeglassesDataset(
        csv_file = './data/files_sample_glasses.csv',
        root_dir = './data/eyeglasses',
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)), # đầu vào cho Resize phải là PIL Image
            # transforms.ToTensor()
        ])
    )

# glass có kích thước 3 x 160 x 160, được normalize về [-1, 1]
eyeglassDataloader = DataLoader(eyeglassesDataset, batch_size=batch_size, shuffle=True)

faceDataset = MeDataset(
        csv_file = './data/landmark.csv',
        root_dir = './data/faces_me',
        transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])
    )

# face có kích thước 3 x 160 x 160, được normalize về [0, 1]
faceDataloader = DataLoader(faceDataset, batch_size=batch_size, shuffle=True)


# ============================================================================
# Training AGN

netG, img_list_glass, img_list_face, G_lossed, D_losses, num_fooled = train_AGN(netG, netD, model_clf, 
                                                                                eyeglassDataloader, faceDataloader, 
                                                                                class_names, nz, num_epochs, batch_size)


torch.save(netG.state_dict(), './model/netG_{}epochs_{}.pth'.format(num_epochs, dt.datetime.today().strftime('%Y%m%d')))




















