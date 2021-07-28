# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 15:21:52 2021

@author: Mai Van Hoa - HUST
"""

from imports import *
from architecture import *


def pretrain_GANs(netD, netG, dataloader, criterion, nz=100, epochs=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0
    beta1 = 0.5
    
    # Setup Adam optimizers for both Generator and Discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-5, betas=(beta1, 0.999))
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    num_epochs = epochs
    
    print('Starting Training ...............')
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            ### Update Discriminator network
            
            # Training with all-real batch
            netD.zero_grad()
            
            data = data.to(device)
            batch_size = data.shape[0]
            # Tạo mảng kích thước (batch_size,) với giá trị real_label
            label = torch.full((batch_size,), real_label, device=device)
            
            # Forward pass real batch through Discriminator
            output = netD(data).view(-1)
            
            # Calculate loss on all_real batch
            errD_real = criterion(output, label) # label phải là kiểu float
            # Calculate gradients for Discriminator in backward pass
            errD_real.backward()
            
            # Training with all_fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with Generator
            fake = netG(noise)
            # điền lại in-place tất cả giá trị trong mảng thành fake_label
            label.fill_(fake_label)
            
            output = netD(fake.detach()).view(-1) # detach đầu ra đang được theo dõi của generator, coi đầu vào cho Descriminator là hằng số
            errD_fake = criterion(output, label)
            
            errD_fake.backward()
            
            errD = (errD_real + errD_fake)/2
            
            # Update Discriminator
            optimizerD.step()
            
            ### Update Generator network
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(real_label)
            output = netD(fake).view(-1)
            
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if i % 10 == 0:
                print('Epoch: {}/{}  | step: {}/{}  | loss G: {} | Loss D: {}'.format(
                        epoch, num_epochs, i, len(dataloader), errG.item(), errD.item()))
                
    return img_list, G_losses, D_losses, netG, netD



if __name__ == '__main__':
    from dataset import *
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    eyeglassesDataset = EyeglassesDataset(
        csv_file = './data/files_sample_glasses.csv',
        root_dir = './data/eyeglasses',
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(160),
            transforms.ToTensor()
        ])
    )
    
    eyeglassesDataloader = DataLoader(eyeglassesDataset, batch_size=64, shuffle=True)
    
    nc, ndf, ngf, nz = 3, 160, 160, 100
    netD = Discriminator(nc=nc, ndf=ndf).to(device)
    netG = Generator(nc=nc, ngf=ngf, nz=nz).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    img_list, G_losses, D_losses, netG, netD = pretrain_GANs(netD, netG, eyeglassesDataloader, criterion)
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
