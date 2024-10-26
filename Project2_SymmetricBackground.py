import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread("icon//Test_Image.png", cv2.IMREAD_GRAYSCALE)

# Bước 2: Tính histogram của ảnh xám
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
hist = hist.ravel()  # Chuyển ma trận thành vector 1D

# Bước 3: Tìm giá trị cực đại (maxp) trong histogram
maxp = np.argmax(hist)

# Bước 4: Tính hàm phân phối xác suất P(a)
P = np.cumsum(hist) / np.sum(hist)

# Bước 5: Tìm giá trị a sao cho P(a) = 95%
p_percent = 0.95  # Tương ứng với 95%
a = np.where(P >= p_percent)[0][0]  # Tìm chỉ số a thỏa mãn P(a) >= 95%

# Bước 6: Tính ngưỡng T dựa trên công thức đối xứng
T = int(maxp - (a - maxp))

# Bước 7: Áp dụng ngưỡng T để phân đoạn ảnh nhị phân
_, binary_image = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc (xám)')

# Ảnh phân đoạn
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title(f'Ảnh phân đoạn (Ngưỡng T = {T})')

plt.show()
