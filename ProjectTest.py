import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk


# Hàm áp dụng ngưỡng đa cấp để phân biệt các vùng xương, phổi, và mô mềm
def multi_threshold_segmentation(image):
    # Ngưỡng cho các vùng xương, phổi, và nền
    ret1, bone = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # Xương (vùng sáng)
    ret2, lung = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)  # Phổi (vùng tối)

    # Kết hợp hai vùng với các phép toán hình thái học
    kernel = np.ones((5, 5), np.uint8)
    bone = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, kernel)  # Làm rõ vùng xương
    lung = cv2.morphologyEx(lung, cv2.MORPH_OPEN, kernel)  # Làm rõ vùng phổi

    combined = cv2.bitwise_or(bone, lung)  # Kết hợp vùng xương và phổi
    return combined, bone, lung


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon//Test_Image.png", cv2.IMREAD_GRAYSCALE)

# Phân đoạn ảnh bằng ngưỡng đa cấp
combined_image, bone_image, lung_image = multi_threshold_segmentation(image)


# Chuyển đổi ảnh sang định dạng phù hợp với tkinter
def cv2_to_tk_image(cv2_image):
    pil_image = Image.fromarray(cv2_image)
    return ImageTk.PhotoImage(pil_image)


# Tạo giao diện với tkinter
root = tk.Tk()
root.title("Phân Đoạn Ảnh X-quang")

# Định kích thước nhỏ hơn cho các ảnh
resize_dim = (300, 240)

# Hiển thị ảnh gốc
image_resized = cv2.resize(image, resize_dim)
image_tk = cv2_to_tk_image(image_resized)
label_image = tk.Label(root, image=image_tk, text="Ảnh Gốc", compound="top", font=("Arial", 12))
label_image.grid(row=0, column=0, padx=5, pady=5)

# Hiển thị ảnh phân đoạn xương
bone_resized = cv2.resize(bone_image, resize_dim)
bone_tk = cv2_to_tk_image(bone_resized)
label_bone = tk.Label(root, image=bone_tk, text="Phân Đoạn Xương", compound="top", font=("Arial", 12))
label_bone.grid(row=0, column=1, padx=5, pady=5)

# Hiển thị ảnh phân đoạn phổi
lung_resized = cv2.resize(lung_image, resize_dim)
lung_tk = cv2_to_tk_image(lung_resized)
label_lung = tk.Label(root, image=lung_tk, text="Phân Đoạn Phổi", compound="top", font=("Arial", 12))
label_lung.grid(row=1, column=0, padx=5, pady=5)

# Hiển thị ảnh phân đoạn kết hợp
combined_resized = cv2.resize(combined_image, resize_dim)
combined_tk = cv2_to_tk_image(combined_resized)
label_combined = tk.Label(root, image=combined_tk, text="Ảnh Phân Đoạn Kết Hợp", compound="top", font=("Arial", 12))
label_combined.grid(row=1, column=1, padx=5, pady=5)

# Bắt đầu vòng lặp chính của tkinter
root.mainloop()
