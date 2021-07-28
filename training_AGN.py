# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:07:17 2021

@author: Mai Van Hoa - HUST
"""

from imports import *
from architecture import *

def train_AGN(netG, netD, model_face, dataloader_glass, dataloader_me, class_names, nz=100, num_epochs=10, batch_size=64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.BCEWithLogitsLoss()
    criterionF = LossF
    
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0
    me_label = [i for i, v in enumerate(class_names) if v == 'Hoa'][0]
    
    beta1 = 0.5
    
    # Setup Adam optimizers for both Generator and Discriminator
    optimizerG = optim.Adam(netG.parameters(), lr=5e-6, betas=(beta1, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=5e-6, betas=(beta1, 0.99))
    
    # Get masks
    imgGlass = cv2.imread('./data/glasses_mask.png', 0)
    r = 160 / imgGlass.shape[1]
    dim = (160, int(imgGlass.shape[0] * r))
    imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
    imgGlass = imgGlass[39:81, 21:138]
    
    imgGlass[imgGlass < 50] = 0
    imgGlass[imgGlass >= 50] = 255
    
    # Tạo mask là mảng nhị phân hai chiều
    mask_Glass = imgGlass/255 # kính trắng, nền đen
    mask_inv_Glass = cv2.bitwise_not(imgGlass)/255 # kính đen, nền trắng
    # plt.imshow(mask_inv_Glass, cmap='gray')
    
    # Lists to keep track of progress
    img_glass_list = [] # mảng lưu trữ kính được tạo ra bởi Generator sau các vòng lặp (giá trị các kính [0, 1])
    img_face_list = []  # là mảng lưu các utils.make_grid                              (giá trị các kính [0, 255])
    G_losses = []
    D_losses = []
    num_fooled = []
    
    print('Starting Training AGN.........')
    for epoch in range(num_epochs):
        for i, data_glass in  tqdm(enumerate(dataloader_glass), total=len(dataloader_glass)):
            # Đủ batch_size để gán mỗi kính cho một khuôn mặt
            if data_glass.shape[0] != batch_size:
                continue
            #=================================================================
            ## Update Discriminator network
            # Train with all-real batch
            netD.zero_grad()
            
            data = data_glass.to(device)
            # batch_size = data.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            
            # Fordward pass real batch through D
            output = netD(data).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            
            # Training with all_fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with Generator
            fake = netG(noise) # fake có kích thước (batch_size, 3, 160, 160)
            
            # điền lại in-place tất cả giá trị trong mảng thành fake_label
            label.fill_(fake_label)
            
            output = netD(fake.detach()).view(-1) # detach đầu ra đang được theo dõi của generator, coi đầu vào cho Descriminator là hằng số
            errD_fake = criterion(output, label)
            
            errD_fake.backward()
            
            errD = (errD_real + errD_fake)/2
            
            # Update Discriminator
            optimizerD.step()
            
            #=================================================================
            ## Check if F(.) is fooled
            # Get a batch of faces and affix glasses to them in correct positions
            # Chú ý: tất cả các bước transformation trên ảnh chỉ có thể sử dụng
            # các hàm của Pytorch để phục vụ quá trình backward
            
            model_face.eval()
            
            # do đầu ra của Generator nằm trong [-1, 1] nên phải normalize về [0, 1]
            fakes = normalize_glass(fake)
            if i % 200 == 0 or (epoch == num_epochs-1 and i == len(dataloader_glass)-1):
                img_glass_list.append(utils.make_grid(fakes.detach(), nrow=8, padding=10)) # 5 CỘT, khoảng cách giữa các hàng/cột là 10
                
            # cắt chỉ lấy phần có kính
            fakes = fakes[:, :, 39:81, 21:138]
            
            for j in range(fakes.size(0)):
                for k in range(fakes.size(1)):
                    fakes[j, k,:,:][mask_Glass == 0] = 0 # đặt phần nền của kính bằng 0
            
            faces, landmarks = next(iter(dataloader_me)) # Chú ý: faces phải có giá trị [0, 1]
            
            # dán kính vào từng khuôn mặt
            for j in range(faces.size(0)):
                face = faces[j, ...]
                # lấy kích thước kính và tọa độ cần chèn tương ứng với khuôn mặt j
                glassWidth, glassHeight = landmarks[j, -2:].int()
                x1, x2, y1, y2 = landmarks[j, :-2].int()
                
                glass = F.interpolate(fakes, (glassHeight, glassWidth)) # resize theo batch, kích thước truyền vào phải là (B x C x H x W)
                mask = F.interpolate(torch.Tensor(mask_Glass[None, None,...]), (glassHeight, glassWidth)).to(device) # None tự động thêm một chiều tương ứng
                mask_inv = F.interpolate(torch.Tensor(mask_inv_Glass[None, None,...]), (glassHeight, glassWidth)).to(device)
            
                # cắt vùng ảnh cần chứa kính trong khuôn mặt để xử lý
                roi = face[None, :, y1:y2, x1:x2]
                roi = roi - mask # mask có kính màu trắng (giá trị 1), nền màu đen (giá trị 0)
                roi = torch.clamp(roi, 0) # vị trí để đặt gọng kính sẽ có giá trị 0
                
                face[:, y1:y2, x1:x2] = glass[j] + roi[0] # glass đang có nền bằng 0 --> chèn gọng kính vào ảnh mặt
                faces[j,...] = face
                
            
            # Tiền xử lý ảnh khuôn mặt trước khi cho vào model_face
            faces = faces * 255
            
            if i % 200 == 0 or (epoch == num_epochs-1 and i == len(dataloader_glass)-1):
                img_face_list.append(utils.make_grid(faces.detach().int(), nrow=8, padding=10))
                
            faces = F.interpolate(faces, (112, 112))
            faces = (faces - 127.5) / 128
            
            # Kiểm tra xem Generator có đánh lừa được model face không
            with torch.no_grad():
                outputs = model_face(faces)
                _, preds = torch.max(outputs, 1)
                if i % 10 == 0:
                    print()
                    print('=====================================')
                    print('Check Generator fooled model face...')
                    print('Epoch: {}/{} | step: {}/{}'.format(epoch+1, num_epochs, i, len(dataloader_glass)))
                    print('Num Fooled: {}/{}'.format(torch.sum(preds != me_label).item(), faces.shape[0]))
                    print('Mean prob me: ', outputs[:, me_label].mean().item())
                    print('=====================================')
                
            
            #=================================================================
            # Update Generator network
            netG.zero_grad()
            label.fill_(real_label)
            
            # Update with Discriminator
            output1 = netD(fake).view(-1)
            l1 = criterion(output1, label)
            l1.backward(retain_graph=True)
            
            # Update with F(.)
            label = torch.full((batch_size,), me_label, device=device)
            output2 = model_face(faces)
            l2 = criterionF(output2, label, device=device, targets=None, type='dodging')
            l2.backward()
            
            errG = l1 + l2
            
            # Update Generator
            optimizerG.step()
            
            
            if i % 10 == 0:
                print()
                print('Epoch: {}/{}  | step: {}/{}  | loss G: {} | Loss D: {}'.format(
                        epoch+1, num_epochs, i, len(dataloader_glass), errG.item(), errD.item()))
                print()
            
    return netG, img_list_glass, img_liss_face, G_lossed, D_losses, num_fooled
    
    
if __name__ == '__main__':
    train_AGN()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    