architecture: chứa kiến trúc của Generator và Discriminator
dataset: khởi tạo Dataset
face_preprocessing: thực hiện các hàm cắt khuôn mặt từ ảnh, tìm vị trí đặt kính
imports: import các thư viện cần thiết
model_inference: khai báo pretrained MobileFaceNet classification
training_AGN: chứa hàm thực hiện training AGN
main: khởi tạo Dataset, DataLoader và các siêu tham số để thực hiện training_AGN
