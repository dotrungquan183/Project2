import cv2
import numpy as np
import matplotlib.pyplot as plt


def top_hat_transform(image, kernel_size=15):
    """ Áp dụng biến đổi top hat. """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat


def bimodal_threshold(image):
    # Bước 1: Tính histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Bước 2: Áp dụng biến đổi top hat
    tophat_image = top_hat_transform(image)

    # Bước 3: Tính histogram của ảnh sau biến đổi top hat
    hist_tophat = cv2.calcHist([tophat_image], [0], None, [256], [0, 256]).flatten()

    # Bước 4: Tìm ngưỡng T
    # Tìm các đỉnh và cực tiểu
    peaks = np.where((hist_tophat[1:-1] > hist_tophat[:-2]) & (hist_tophat[1:-1] > hist_tophat[2:]))[0] + 1

    if len(peaks) >= 2:
        # Giả định rằng hai đỉnh lớn nhất là các đối tượng
        max_peaks = sorted(peaks, key=lambda x: hist_tophat[x], reverse=True)[:2]
        T = (max_peaks[0] + max_peaks[1]) // 2  # Ngưỡng là trung bình của hai đỉnh lớn nhất
    else:
        T = np.argmax(hist)  # Nếu không tìm thấy đỉnh, dùng ngưỡng mặc định

    return T, hist, hist_tophat


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon//Test_Image.png", cv2.IMREAD_GRAYSCALE)

# Tính ngưỡng theo histogram bimodal
threshold, histogram, histogram_tophat = bimodal_threshold(image)
print(f'Ngưỡng cuối cùng: {threshold}')

# Phân đoạn ảnh sử dụng ngưỡng tìm được
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Hiển thị histogram và ảnh đã phân đoạn
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Histogram')
plt.plot(histogram, label='Original Histogram')
plt.plot(histogram_tophat, label='Top Hat Histogram')
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
