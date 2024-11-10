import cv2
import matplotlib.pyplot as plt

# Đọc ảnh PNG
image = cv2.imread("icon/PhongCanh.jpg", cv2.IMREAD_GRAYSCALE)

# Áp dụng Otsu Thresholding
_, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Hiển thị ảnh gốc và ảnh phân đoạn
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Ảnh Gốc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap="gray")
plt.title("Ảnh Phân Đoạn với Otsu")
plt.axis("off")

plt.show()
