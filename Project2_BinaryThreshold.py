import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Đường dẫn tới ảnh (có thể thay thế với đường dẫn tuyệt đối nếu ảnh nằm trong thư mục của bạn)
img_path = 'icon/Test_Image.png'

# Kiểm tra xem ảnh có tồn tại không
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found at: {img_path}")

# Đọc ảnh bằng OpenCV
img = cv2.imread(img_path)

# Kiểm tra xem ảnh có được đọc thành công không
if img is None:
    raise ValueError("Failed to load image. Please check the file path.")

# Chuyển ảnh thành ảnh xám
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Lọc ảnh với ngưỡng 127
ret, th_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Hiển thị ảnh
fg, ax = plt.subplots(1, 2, figsize=(8, 4))
for i, image in enumerate([img_gray, th_img]):
    ax[i].imshow(image, cmap='gray')  # Thêm cmap='gray' để hiển thị ảnh xám đúng cách
    ax[i].axis('off')  # Tắt trục
    if i == 0:
        ax[i].set_title('Original Image')
    else:
        ax[i].set_title('Threshold Image')

# Hiển thị biểu đồ
plt.show()
