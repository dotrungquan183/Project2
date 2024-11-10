import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, mark_boundaries
import os

# Đường dẫn tới ảnh
img_path = 'icon/Test_Image.png'

# Kiểm tra xem ảnh có tồn tại không
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found at: {img_path}")

# Đọc ảnh bằng OpenCV
img = cv2.imread(img_path)

# Kiểm tra xem ảnh có được đọc thành công không
if img is None:
    raise ValueError("Failed to load image. Please check the file path.")

# Chuyển ảnh từ BGR sang RGB (OpenCV mặc định là BGR, Matplotlib sử dụng RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Áp dụng thuật toán Quickshift để phân đoạn ảnh
segments_quick = quickshift(img_rgb, kernel_size=5, max_dist=10, ratio=0.5)

# Hiển thị ảnh gốc và kết quả phân đoạn
fg, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')  # Tắt trục

# Hiển thị kết quả phân đoạn với đường biên
ax[1].imshow(mark_boundaries(img_rgb, segments_quick))
ax[1].set_title('Quickshift Segmentation')
ax[1].axis('off')  # Tắt trục

# Hiển thị biểu đồ
plt.show()
