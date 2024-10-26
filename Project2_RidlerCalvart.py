import cv2
import numpy as np

# Hàm để tính ngưỡng theo thuật toán đẳng liệu
def ridler_calvard_thresholding(image, max_iterations=1000, tolerance=1e-5):
    # B1: Chọn giá trị ngưỡng ban đầu (lấy trung bình của tất cả các giá trị pixel)
    theta_prev = np.mean(image)

    for i in range(max_iterations):
        # Chia ảnh thành hai nhóm: đối tượng và nền dựa trên ngưỡng hiện tại
        foreground = image[image > theta_prev]
        background = image[image <= theta_prev]

        # B2: Tính trung bình của hai nhóm
        m_f = np.mean(foreground) if len(foreground) > 0 else 0
        m_b = np.mean(background) if len(background) > 0 else 0

        # B3: Tính ngưỡng mới
        theta_new = (m_f + m_b) / 2

        # B4: Kiểm tra điều kiện dừng
        if abs(theta_new - theta_prev) < tolerance:
            break

        # Cập nhật ngưỡng để lặp tiếp
        theta_prev = theta_new

    return theta_new


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon//Test_Image.png", cv2.IMREAD_GRAYSCALE)

# Áp dụng thuật toán đẳng liệu để tìm ngưỡng
threshold = ridler_calvard_thresholding(image)
print(f'Ngưỡng cuối cùng: {threshold}')

# Phân đoạn ảnh sử dụng ngưỡng tìm được
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Hiển thị chỉ ảnh đã phân đoạn
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
