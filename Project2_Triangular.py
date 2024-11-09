from tkinter import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Hàm để chuyển đổi ảnh OpenCV sang ảnh Tkinter
def cv2_to_tk_image(cv_img):
    color_converted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    return ImageTk.PhotoImage(pil_img)


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
    for b in range(30, 256):
        Hb = hist[b]
        if Hb > 8000:  # Chỉ tính khoảng cách khi Hb lớn hơn 3
            # Công thức tính khoảng cách từ điểm đến đường thẳng
            d = abs((y2 - y1) * b - (x2 - x1) * Hb + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(d)
        else:
            distances.append(0)

    # Tìm mức xám b tại ngưỡng T
    T = np.argmax(distances)

    return T, hist



# Hàm để hiển thị histogram trong giao diện Tkinter
def display_histogram(canvas_frame, hist, threshold):
    # Tạo Figure và vẽ histogram với kích thước mong muốn
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(hist, color='black')
    ax.axvline(x=threshold, color='r', linestyle='--', label='Ngưỡng T')
    ax.set_xlabel('Mức xám', fontsize=10)
    ax.set_ylabel('Tần số', fontsize=10)
    ax.set_title('Histogram của ảnh xám', fontsize=12)
    ax.legend()
    ax.grid(True)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)

    # Hiển thị Figure trên Canvas Tkinter
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()

    # Đặt Canvas vào giao diện Tkinter
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.config(width=800, height=400)
    canvas_widget.place(x=40, y=40)


# Đọc ảnh và chuyển thành ảnh xám
image = cv2.imread("icon//Test_Image.png", cv2.IMREAD_GRAYSCALE)

# Áp dụng thuật toán tam giác để tìm ngưỡng
threshold, histogram = triangle_thresholding(image)

# Phân đoạn ảnh sử dụng ngưỡng tìm được
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Tạo giao diện Tkinter
root = Tk()
root.title("Thuật toán ngưỡng tam giác")
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

# Hiển thị histogram ở góc trên bên phải
frame_histogram = Frame(root)
frame_histogram.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
display_histogram(frame_histogram, histogram, threshold)

# Hiển thị ảnh phân đoạn ở góc dưới bên trái
segmented_resized = cv2.resize(segmented_image, (700, 400))
segmented_tk = cv2_to_tk_image(cv2.cvtColor(segmented_resized, cv2.COLOR_GRAY2BGR))
label_segmented = Label(root, image=segmented_tk, text=f"Ảnh phân đoạn (Ngưỡng T = {threshold})",
                        compound="top", font=("Arial", 14))
label_segmented.place(x=100, y=500)

# Khung chú thích
frame_legend = Frame(root, bd=1, relief="groove", padx=15, pady=15)
frame_legend.place(x=1000, y=500)

# Thêm tiêu đề "Chú thích" vào khung chú thích
title_label = Label(frame_legend, text="Chú thích", font=("Arial", 14, "bold"), anchor="w")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

# Căn giữa các thành phần trong khung chú thích
frame_legend.grid_columnconfigure(0, weight=1)
frame_legend.grid_columnconfigure(1, weight=1)

# Hộp đen cho đối tượng và nhãn
black_box = Label(frame_legend, bg="black", width=5, height=2)
black_box.grid(row=1, column=0, padx=10, pady=10)
label_black = Label(frame_legend, text="Đối tượng", font=("Arial", 12), anchor="w")
label_black.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Hộp trắng cho nền và nhãn
white_box = Label(frame_legend, bg="white", width=5, height=2)
white_box.grid(row=2, column=0, padx=10, pady=10)
label_white = Label(frame_legend, text="Nền", font=("Arial", 12), anchor="w")
label_white.grid(row=2, column=1, padx=10, pady=10, sticky="w")

# Khởi chạy vòng lặp chính của Tkinter
root.mainloop()