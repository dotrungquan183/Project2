import cv2
import numpy as np
import matplotlib.pyplot as plt


# Hàm để tính ngưỡng theo thuật toán tam giác
def triangle_thresholding(image):
    # Tính histogram của ảnh
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Tìm Hmax và Hmin
    bmax = np.argmax(hist)  # Mức xám có histogram lớn nhất
    Hmax = hist[bmax]
    bmin = np.argmin(hist)  # Mức xám có histogram nhỏ nhất
    Hmin = hist[bmin]

    # Xây dựng đường thẳng A
    x1, y1 = bmax, Hmax
    x2, y2 = bmin, Hmin

    # Tính khoảng cách từ mỗi điểm đến đường thẳng
    distances = []
    for b in range(256):
        Hb = hist[b]
        if Hb > 0:
            # Công thức tính khoảng cách từ điểm đến đường thẳng
            d = abs((y2 - y1) * b - (x2 - x1) * Hb + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(d)
        else:
            distances.append(0)

    # Tìm mức xám b tại ngưỡng T
    T = np.argmax(distances)

    return T, hist


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon//Test_Image.png", cv2.IMREAD_GRAYSCALE)

# Áp dụng thuật toán tam giác để tìm ngưỡng
threshold, histogram = triangle_thresholding(image)
print(f'Ngưỡng cuối cùng: {threshold}')

# Phân đoạn ảnh sử dụng ngưỡng tìm được
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Hiển thị histogram và ảnh đã phân đoạn
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Histogram')
plt.plot(histogram)
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
