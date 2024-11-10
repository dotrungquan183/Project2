import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def cv2_to_tk_image(cv_img):
    """Chuyển đổi ảnh OpenCV sang ảnh Tkinter"""
    color_converted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    return ImageTk.PhotoImage(pil_img)


def top_hat_transform(image, kernel_size=15):
    """Áp dụng biến đổi top hat"""
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
        T = (max_peaks[0] + max_peaks[1])/2  # Ngưỡng là trung bình của hai đỉnh lớn nhất
    else:
        T = np.argmax(hist)  # Nếu không tìm thấy đỉnh, dùng ngưỡng mặc định

    return T, hist, hist_tophat


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon//NaoBo.png", cv2.IMREAD_GRAYSCALE)

# Tính ngưỡng theo histogram bimodal
threshold, histogram, histogram_tophat = bimodal_threshold(image)
print(f'Ngưỡng cuối cùng: {threshold}')

# Phân đoạn ảnh sử dụng ngưỡng tìm được
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Tạo giao diện Tkinter
root = Tk()
root.title("Biến đổi Top Hat và Phân đoạn Ảnh")
root.state("zoomed")

# Đặt trọng số hàng và cột cho căn giữa các thành phần
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Hiển thị ảnh gốc ở góc trên bên trái
image_resized = cv2.resize(image, (700, 400))
image_tk = cv2_to_tk_image(cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR))
label_image = Label(root, image=image_tk, text="Ảnh gốc", compound="top", font=("Arial", 14))
label_image.place(x=100, y=50)

# Hiển thị histogram và ảnh đã phân đoạn trong giao diện Tkinter
frame_histogram = Frame(root)
frame_histogram.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Tạo figure cho histogram
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(histogram, label='Original Histogram')
ax.plot(histogram_tophat, label='Top Hat Histogram')
ax.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
ax.set_xlabel('Mức xám', fontsize=10)
ax.set_ylabel('Tần số', fontsize=10)
ax.set_title('Histogram của ảnh xám', fontsize=12)
ax.legend()

# Hiển thị figure trên canvas
canvas = FigureCanvasTkAgg(fig, master=frame_histogram)
canvas.draw()

# Đặt Canvas vào giao diện Tkinter
canvas_widget = canvas.get_tk_widget()
canvas_widget.config(width=800, height=400)
canvas_widget.place(x=40, y=40)

# Hiển thị ảnh phân đoạn ở góc dưới bên trái
segmented_resized = cv2.resize(segmented_image, (700, 400))
segmented_tk = cv2_to_tk_image(cv2.cvtColor(segmented_resized, cv2.COLOR_GRAY2BGR))
label_segmented = Label(root, image=segmented_tk, text=f"Ảnh phân đoạn (Ngưỡng T = {threshold})",
                        compound="top", font=("Arial", 14))
label_segmented.place(x=100, y=500)

# Tạo bảng chú thích
frame_legend = Frame(root, bd=1, relief="groove", padx=15, pady=15)
frame_legend.place(x=1000, y=500)

# Thêm tiêu đề "Chú thích" vào đầu bảng chú thích
title_label = Label(frame_legend, text="Chú thích", font=("Arial", 14, "bold"), anchor="w")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

# Căn giữa các thành phần bên trong bảng chú thích
frame_legend.grid_columnconfigure(0, weight=1)
frame_legend.grid_columnconfigure(1, weight=1)

# Ô màu đen cho đối tượng và chú thích
black_box = Label(frame_legend, bg="black", width=5, height=2)
black_box.grid(row=1, column=0, padx=10, pady=10)
label_black = Label(frame_legend, text="Đối tượng", font=("Arial", 12), anchor="w")
label_black.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Ô màu trắng cho nền và chú thích
white_box = Label(frame_legend, bg="white", width=5, height=2)
white_box.grid(row=2, column=0, padx=10, pady=10)
label_white = Label(frame_legend, text="Nền", font=("Arial", 12), anchor="w")
label_white.grid(row=2, column=1, padx=10, pady=10, sticky="w")

# Khởi chạy vòng lặp chính của Tkinter
root.mainloop()
