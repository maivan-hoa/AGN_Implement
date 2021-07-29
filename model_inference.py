# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:59:34 2021

@author: Mai Van Hoa - HUST
"""
from imports import *
from check_affix_glass import *


class Model_classification():
    def __init__(self, model_xml, model_bin):
        self.model_xml = model_xml
        self.model_bin = model_bin
        ie = IECore()
        net =  ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        self.exec_net =  ie.load_network(network=net, device_name="CPU")
        
        
    def predict(self, input):
        result_vector = self.exec_net.infer(inputs={self.input_blob: input})
        result_vector = result_vector[self.out_blob]
        return result_vector[0]
    
    
class Model_detector():
    def __init__(self, model_xml, model_bin):
        self.model_xml = model_xml
        self.model_bin = model_bin
        ie = IECore()
        net =  ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        self.exec_net =  ie.load_network(network=net, device_name="CPU")
        self.n, self.c, self.h, self.w = net.input_info[self.input_blob].input_data.shape
        
    
    def predict(self, input, conf=0.5):
        h_origin, w_origin, _ = input.shape
        corr_faces = []
        input = cv2.resize(input, (self.w, self.h))
        input = input.transpose((2, 0, 1))
        input = input[np.newaxis, ...]
        results = self.exec_net.infer(inputs={self.input_blob: input})
        results = results[self.out_blob]
        
        for res in results[0, 0]:
            if res[2] > conf:
                # kết quả là tỷ lệ tọa độ khung hình đầu vào
                x_min = max(res[3], 0)
                y_min = max(res[4], 0)
                x_max = max(res[5], 0)
                y_max = max(res[6], 0)
                
                # convert sang tọa độ của ảnh gốc ban đầu
                x1 = int(x_min * w_origin)
                y1 = int(y_min * h_origin)
                x2 = int(x_max * w_origin)
                y2 = int(y_max * h_origin)
                corr_faces.append((x1, y1, x2, y2))
        return corr_faces
    

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = nn.Sequential(
            BatchNorm1d(embedding_size),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        
        return out
    

def get_MobileFaceNet_classification(path):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    
    model = MobileFaceNet(512).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    
    print()
    print('Get Model Success.........')
    return model.to(device)
    

if __name__ == '__main__':
    
    CLASSES = {
        0: 'Bien',
        1: 'Cuong',
        2: 'LeDong',
        3: 'Phu',
        4: 'Vu',
        5: 'Nguyen',
        6: 'Hoa'
    }
    
    path_image = r'C:\Users\DELL\Desktop\New folder\AGN-Implement\data\faces_me\hoa_57.png'
    path_glass = r"C:\Users\DELL\Desktop\New folder\AGN-Implement\data/eyeglasses/glasses000002-2.png"
    
    path = './model/MobileFaceNet_classification.pth'
    
    #==============================
    # ảnh khuôn mặt ban đầu
    face = io.imread(path_image)
    #==============================
    # ảnh mặt có dán kính
    face = affix_glass(path_image, path_glass).permute(1,2,0).numpy().astype('uint8')
    #===============================
    
    face = cv2.resize(face, (112, 112))
    face = face.transpose((2, 0, 1))
    face = torch.Tensor(face)
    face = (face - 127.5) / 128
    
    model = get_MobileFaceNet_classification(path)
    
    model.eval()
    with torch.no_grad():
        output = model(face[None,...])
    
    cl = np.argmax(output[0]).item()
    print('Output: ', output[0])
    print('Predicted: ', CLASSES[cl])
    









    
    
    