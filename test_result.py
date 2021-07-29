# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:21:09 2021

@author: Mai Van Hoa - HUST
"""
from imports import *
from architecture import *
from model_inference import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nc, ndf, ngf, nz = 3, 160, 160, 100

CLASSES = {
    0: 'Bien',
    1: 'Cuong',
    2: 'LeDong',
    3: 'Phu',
    4: 'Vu',
    5: 'Nguyen',
    6: 'Hoa'
}


imgGlass = cv2.imread('./data/glasses_mask.png', 0)
r = 160 / imgGlass.shape[1]
dim = (160, int(imgGlass.shape[0] * r))
imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
imgGlass = imgGlass[39:81, 21:138]

# height và width ban đầu của kính
origGlassHeight, origGlassWidth = imgGlass.shape[:2]

imgGlass[imgGlass < 50] = 0
imgGlass[imgGlass >= 50] = 255
# plt.imshow(imgGlass)

# Tạo mask là mảng nhị phân hai chiều
mask_Glass = imgGlass/255 # kính trắng, nền đen


def generate_glass(path_model):
    '''
    Trả về kính có kích thước 160 x 160 x 3
    Giá trị kính nằm trong [0, 1]
    '''
    netG = Generator(nc=nc, ngf=ngf, nz=nz).to(device)
    netG.load_state_dict(torch.load(path_model, map_location=device))
    
    netG.eval()
    
    with torch.no_grad():
        noise = torch.randn(1, 100, 1, 1, device=device)
        glass = netG(noise)
    
    glass = normalize_glass(glass)
    return glass[0].permute(1,2,0)


def affix_glass(path_model_generator, path_image):
    '''
    Trả về ảnh khuôn mặt gốc và ảnh khuôn mặt có dán thêm kính trong hệ màu RGB
    Giá trị nằm trong [0, 255]
    '''
    glass_fake = generate_glass(path_model_generator).numpy()
    plt.imshow(glass_fake)
    plt.show()
    
    glass_fake = glass_fake[39:81, 21:138]
    glass_fake[mask_Glass == 0] = 0
    
    # Ảnh BGR
    image = cv2.imread(path_image)
    corr_face = model_detector.predict(image)[0]
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    a1, b1, a2, b2 = corr_face
    drect = dlib.rectangle(a1, b1, a2, b2)
    
    face_origin = image[b1:b2, a1:a2].astype('uint8')
    # detect landmarks
    shapes = predict_landmark(image, drect)
    
    # Tìm height và width của kính trong ảnh khuôn mặt
    glassWidth = shapes.part(16).x - shapes.part(0).x
    glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)
    
    # tìm vùng để đặt kính
    y1 = int(shapes.part(22).y)
    y2 = int(y1 + glassHeight)
    x1 = int(shapes.part(27).x - glassWidth/2)
    x2 = x1 + glassWidth
    
    # Xử lý khi tọa độ các landmark vượt ra ngoài kích thước khung ảnh
    if y1 < 0:
        y1 = 0
        glassHeight -= abs(y1)
    if x1 < 0:
        glassWidth -= abs(x1)
        x1 = 0
    if x2 > image.shape[1]:
        glassWidth -= (x2 - image.shape[1])
        x2 = image.shape[1]
        
    glass = cv2.resize(glass_fake, (glassWidth, glassHeight))
    mask = cv2.resize(mask_Glass, (glassWidth, glassHeight))
    mask = np.stack((mask,)*3, -1)
    
    image = image/255
    roi = image[y1:y2, x1:x2]
    roi = roi - mask
    roi = np.clip(roi, 0, 1)
    image[y1:y2, x1:x2] = glass + roi 
    
    face_affix_glass = (image[b1:b2, a1:a2]*255).astype('uint8')
    
    return face_origin, face_affix_glass


def predict_class_face(face, model_clf, required_size=(112, 112)):
    
    # resize face to the model size
    face = cv2.resize(face, required_size)
    
    face = np.asarray([face])
    # prepare the face for the model
    face = face.transpose((0, 3, 1, 2))
    face = (face - 127.5) / 128
    
    face = torch.Tensor(face)
    model_clf.eval()
    with torch.no_grad():
        # perform prediction
        yhat = model_clf(face)
        softmax = nn.Softmax(dim=1)
        prob = softmax(yhat)
        
    cl = np.argmax(yhat[0]).item()
    
    return CLASSES[cl], prob[0][cl], prob[0].numpy()
    
    
    
if __name__ == '__main__':
    
    # Yêu cầu hệ màu đầu vào là BGR
    model_detector_xml = './model/face-detection-0204_fp16.xml'
    model_detector_bin = './model/face-detection-0204_fp16.bin'
    model_detector = Model_detector(model_detector_xml, model_detector_bin)
    
    predict_landmark = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
    
    # Khai báo đường dẫn đến các pre-trained
    path_model_generator = './model/netG_backup.pth'
    path_model_classification = './model/MobileFaceNet_classification.pth'
    model_clf = get_MobileFaceNet_classification(path_model_classification)
    
    # Đường dẫn đến ảnh cần test
    path_image = r'.\data\faces_me\hoa_57.png'
    
    # face_origin = io.imread(path_image)
    face_origin, face_affix_glass = affix_glass(path_model_generator, path_image)
    
    class_org, prob_org, prob_all = predict_class_face(face_origin, model_clf)
    print('Image origin: class: {}, probability: {}'.format(class_org, prob_org))
    print('Probability all: ', np.round(prob_all, decimals=3))
    
    print('======================================================')
    class_dod, prob_dod, prob_all = predict_class_face(face_affix_glass, model_clf)
    print('Image affix glass: class: {}, probability: {}'.format(class_dod, prob_dod))
    print('Probability all: ', np.round(prob_all, 3))
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(face_origin)
    ax[0].axis('off')
    
    ax[1].imshow(face_affix_glass)
    ax[1].axis('off')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    