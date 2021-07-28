# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:54:52 2021

@author: Mai Van Hoa - HUST
"""

from imports import *

class Generator(nn.Module):
    def __init__(self, nc=3, ngf=160, nz=100):
        '''
        Parameters
        ----------
        nc : Số kênh đầu ra của ảnh được tạo bởi generator
        ngf : số kênh cơ sở của các lớp tích chập
        nz : số chiều của nhiễu đầu vào

        Returns
        -------
        Ảnh giả mạo được tạo từ nhiễu đầu vào Z

        '''
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # in_channels, out_channnels, kernel_size, stride, padding
            
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(inplace=True/False)
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 3, 3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 43 x 43
            nn.ConvTranspose2d( ngf, nc, 4, 4, 6, bias=False),
            nn.Tanh()

            # kích thước đầu ra
            # state size. (nc) x 160 x 160
        )
        
    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=160):
        '''
        Nhận đầu vào là ảnh được tạo ra từ Generator -> kiểm tra xem là ảnh
        thật hay giả

        '''
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            # input is (nc) x 160 x 160
            nn.Conv2d(nc, ndf, 4, 4, 6, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf) x 43 x 43
            nn.Conv2d(ndf, ndf * 2, 4, 3, 3, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
            
            # kích thước đầu ra 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)



def LossF(logits, labels, targets=None, type='dodging'):
    '''
    Parameters
    ----------
    logits : phân phối xác suất được trả về bởi model face

    label : nhãn của batch dữ liệu

    target : Nếu type='impersonate' thì cần phải truyền vào chỉ số lớp cần mạo danh 

    type : 'dodging' or 'impersonate'
    '''
    
    
    if type == 'dodging':
        label_one_hot = torch.eye(logits.shape[1])[labels.long()]
        
        real = torch.sum(logits * label_one_hot, 1)
        other = torch.sum(logits * (1 - label_one_hot), 1)
        
        return torch.mean(real - other)
    
    if type == 'impersonate':
        target_one_hot = torch.eye(logits.shape[1])[targets.long()]
        
        target = torch.sum(logits * target_one_hot, 1)
        other = torch.sum(logits * (1 - target_one_hot), 1)
        
        return torch.mean(other - target)
        
    
def normalize_glass(fake):
    return (fake - fake.min()) / (fake.max() - fake.min())



if __name__ == '__main__':
    from torchsummary import summary
    
    # print('Generator:')
    # model = Generator()
    # summary(model, (100, 1, 1))
    
    # print('Discriminator')
    # model = Discriminator()
    # summary(model, (3,160,160))
    
    
    logits = torch.FloatTensor([[1,2,3], [4,5,6]])
    labels = torch.FloatTensor([1,1])
    
    print(LossF(logits, labels))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


