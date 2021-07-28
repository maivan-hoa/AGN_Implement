# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 21:15:30 2021

@author: Mai Van Hoa - HUST
"""
from imports import *

def get_image_camera(path_save):
    for file in os.listdir(path_save):
        os.remove(os.path.join(path_save, file))
    
    camera = cv2.VideoCapture(0)
    count = 0
    
    while True:
        success, image = camera.read()
        cv2.imwrite(path_save + '/hoa_{}.png'.format(count), image)
        time.sleep(0.2)
        count += 1
        
        cv2.imshow('frame', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()
    

def crop_face(path_img, path_save_face):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./model/mmod_human_face_detector.dat')
    
    for path in os.listdir(path_img):
        img = io.imread(os.path.join(path_img, path))
        det = cnn_face_detector(img)[0]
        x1, y1, x2, y2 = det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()
        face = img[y1:y2, x1:x2]
        
        face = cv2.resize(face, (160, 160), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(path_save_face, path), cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        

def save_landmark(path_face, path_file_csv):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./model/mmod_human_face_detector.dat')
    predict_landmark = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
    
    # Get original glass height and width for calculations
    imgGlass = cv2.imread("data/eyeglasses/glasses000002-2.png", -1)
    r = 160.0 / imgGlass.shape[1]
    dim = (160, int(imgGlass.shape[0] * r))

    # perform the actual resizing of the image and show it
    imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
    imgGlass = imgGlass[39:81, 21:138]
    # height và width ban đầu của kính
    origGlassHeight, origGlassWidth = imgGlass.shape[:2]
    
    with open(path_file_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        
        for path in os.listdir(path_face):
            face = io.imread(os.path.join(path_face, path))
            det = cnn_face_detector(face)[0]
            shapes = predict_landmark(face, det.rect)
            
            # for i in range(68):
            #     x, y = shapes.part(i).x, shapes.part(i).y
            #     cv2.circle(face, (x, y), 1, (0, 0, 255), -1)
                
            # plt.imshow(face)
        
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
            if x2 > face.shape[1]:
                glassWidth -= (x2 - face.shape[1])
                x2 = face.shape[1]
                
            writer.writerow([path, x1, x2, y1, y2, glassWidth, glassHeight])
            # imgGlass = cv2.resize(imgGlass, (glassWidth, glassHeight), interpolation = cv2.INTER_AREA)
            # face[y1:y2, x1:x2] = imgGlass
            # plt.imshow(face)
        
        
            
    
if __name__ == '__main__':
    path_img = './data/images_me'
    # get_image_camera(path_img)
    
    
    path_save_face = './data/faces_me'
    path_file_csv = './data/landmark.csv'
    
    # crop_face(path_img, path_save_face)
    
    save_landmark(path_save_face, path_file_csv)
    
    
    
    
    
    
    
    
    
    
    
    