# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:48:33 2021

@author: Mai Van Hoa - HUST
"""

from imports import *
from model_inference import *

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


def predict_class_face(face, model_clf, required_size=(112, 112)):
    
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # resize face to the model size
    face = cv2.resize(face, required_size)
    
    face = np.asarray([face], dtype='float16')
    # prepare the face for the model
    face = face.transpose((0, 3, 1, 2))
    face = (face - 127.5) / 128
    
    # perform prediction
    yhat = model_clf.predict(face)
    cl = np.argmax(yhat)
    
    return CLASSES[cl]


def affix_glass(image, glass_fake, predict_landmark, drect):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    
    image = (image*255).astype('uint8')
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # (image*255).astype('uint8')


def run(netG, model_clf, model_detector, predict_landmark, required_size=(112, 112)):
    
    # Kính để test
    glass = io.imread("./data/eyeglasses/glasses000002-2.png")
    glass = cv2.resize(glass, (160, 160), interpolation = cv2.INTER_AREA)
    plt.imshow(glass)
    glass = glass/255
    # glass = np.transpose(glass, (2, 0, 1))
    glass = glass[39:81, 21:138]

    camera = cv2.VideoCapture(0)
    
    while True:
        success, image = camera.read()

        corr_faces = model_detector.predict(image)
        for corr in corr_faces:
            # left, top, right, bottom
            x1, y1, x2, y2 = corr
            drect = dlib.rectangle(x1, y1, x2, y2)
            
            
            # for i in range(68):
            #     x, y = shapes.part(i).x, shapes.part(i).y
            #     cv2.circle(image, (x, y), 1, (0, 225, 0), -1)
            
            # classification face
            # face = image[y1:y2, x1:x2]
            # predicted = predict_class_face(face, model_clf)
            
            image = affix_glass(image, glass, predict_landmark, drect)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(image, predicted, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    
            
            
        cv2.imshow('frame', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()
    
    

if __name__ == '__main__':
    required_size=(112, 112)
    
    netG = 'abc'
    
    model_clf_xml = './model/MobileFaceNet_classification.xml'
    model_clf_bin = './model/MobileFaceNet_classification.bin'
    model_clf = Model_classification(model_clf_xml, model_clf_bin)
    
    # Yêu cầu hệ màu đầu vào là BGR
    model_detector_xml = './model/face-detection-0204_fp16.xml'
    model_detector_bin = './model/face-detection-0204_fp16.bin'
    model_detector = Model_detector(model_detector_xml, model_detector_bin)
    
    predict_landmark = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
    
    run(netG, model_clf, model_detector, predict_landmark)















