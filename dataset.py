# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:30:58 2021

@author: Mai Van Hoa - HUST
"""
from imports import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EyeglassesDataset(Dataset):
    """Eyeglasses dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pd.read_csv(csv_file) # file csv chứa tên các ảnh
        self.root_dir = root_dir # thư mục chứa ảnh
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.label.iloc[idx, 0])
        image = io.imread(img_name) # Ảnh phải đọc ra kiểu dữ liệu uint8
        sample = image

        if self.transform:
            sample = self.transform(sample)
            sample = np.asarray(sample)
        
        sample = (sample - 127.5) / 128 # normalize giá trị về [-1, 1]
        sample = torch.Tensor(sample).permute(2,0,1)
        return sample.to(device)
        
    
class MeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform_img=None):
        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_img = transform_img
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label.iloc[idx, 0])
        image = io.imread(img_name)
        
        if self.transform_img:
            image = self.transform_img(image)
        
        landmarks = self.label.iloc[idx, 1:]
        
        return image.to(device), torch.Tensor(landmarks).to(device) # Yêu cầu kết quả trả về phải là tensor để tạo batch
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    