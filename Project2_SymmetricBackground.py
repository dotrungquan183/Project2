from tkinter import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Hàm chuyển đổi ảnh OpenCV sang ảnh Tkinter
def cv2_to_tk_image(cv_img):
    color_converted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    return ImageTk.PhotoImage(pil_img)


def display_histogram(canvas_frame, image):
    # Tạo biểu đồ và vẽ biểu đồ histogram với kích thước mong muốn
    fig, ax = plt.subplots(figsize=(6, 2))  # Điều chỉnh kích thước histogram
    ax.hist(image.ravel(), bins=256, color='black')
    ax.set_xlabel('Cấp độ xám', fontsize=10)
    ax.set_ylabel('Tần số', fontsize=10)
    ax.set_title('Histogram của ảnh xám', fontsize=12)
    ax.grid(True)

    # Điều chỉnh khoảng cách bên trong của biểu đồ
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)

    # Hiển thị biểu đồ trên Canvas của Tkinter
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()

    # Đặt widget canvas ở giữa với các tùy chọn căn chỉnh và mở rộng
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.config(width=800, height=400)  # Điều chỉnh kích thước canvas nếu cần
    canvas_widget.place(x=40, y=40)


# Bước 1: Tải ảnh và chuyển sang ảnh xám
image = cv2.imread("icon//NaoBo.png", cv2.IMREAD_GRAYSCALE)

# Bước 2: Tính toán histogram của ảnh xám
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
hist = hist.ravel()  # Chuyển ma trận thành vector 1D

# Bước 3: Tìm giá trị đỉnh (maxp) trong histogram
maxp = np.argmax(hist)

# Bước 4: Tính hàm phân phối xác suất P(a)
P = np.cumsum(hist) / np.sum(hist)

# Bước 5: Tìm giá trị của 'a' sao cho P(a) = 95%
p_percent = 0.95  # Tương đương với 95%
a = np.where(P >= p_percent)[0][0]  # Tìm chỉ số 'a' thỏa mãn P(a) >= 95%

# Bước 6: Tính ngưỡng T dựa trên công thức đối xứng
T = int(maxp - (a - maxp))

# Bước 7: Áp dụng ngưỡng T để phân đoạn nhị phân
_, binary_image = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

# Tạo giao diện Tkinter
root = Tk()
root.title("Thuật toán đối xứng nền")
root.state("zoomed")  # Chế độ toàn màn hình

# Đặt trọng số hàng và cột để căn giữa các phần tử
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Hiển thị ảnh gốc ở góc trên bên trái
image_resized = cv2.resize(image, (700, 400))
image_tk = cv2_to_tk_image(cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR))
label_image = Label(root, image=image_tk, text="Ảnh gốc", compound="top", font=("Arial", 14))
label_image.place(x=100, y=50)

# Hiển thị histogram ở góc trên bên phải
frame_histogram = Frame(root)
frame_histogram.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
display_histogram(frame_histogram, image)

# Hiển thị ảnh đã phân đoạn ở góc dưới bên trái
segmented_resized = cv2.resize(binary_image, (700, 400))
segmented_tk = cv2_to_tk_image(cv2.cvtColor(segmented_resized, cv2.COLOR_GRAY2BGR))
label_segmented = Label(root, image=segmented_tk, text=f"Ảnh phân đoạn (Ngưỡng T = {T})",
                        compound="top", font=("Arial", 14))
label_segmented.place(x=100, y=500)

# Tạo khung chú thích
frame_legend = Frame(root, bd=1, relief="groove", padx=15, pady=15)
frame_legend.place(x=1000, y=500)

# Thêm tiêu đề "Chú thích" lên đầu khung chú thích
title_label = Label(frame_legend, text="Chú thích", font=("Arial", 14, "bold"), anchor="w")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

# Căn giữa các phần tử bên trong khung chú thích
frame_legend.grid_columnconfigure(0, weight=1)
frame_legend.grid_columnconfigure(1, weight=1)

# Hộp màu đen cho đối tượng và nhãn
black_box = Label(frame_legend, bg="black", width=5, height=2)
black_box.grid(row=1, column=0, padx=10, pady=10)
label_black = Label(frame_legend, text="Đối tượng", font=("Arial", 12), anchor="w")
label_black.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Hộp màu trắng cho nền và nhãn
white_box = Label(frame_legend, bg="white", width=5, height=2)
white_box.grid(row=2, column=0, padx=10, pady=10)
label_white = Label(frame_legend, text="Nền", font=("Arial", 12), anchor="w")
label_white.grid(row=2, column=1, padx=10, pady=10, sticky="w")

# Khởi động vòng lặp chính của Tkinter
root.mainloop()
